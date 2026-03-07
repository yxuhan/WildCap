import argparse
import os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# loss configs
parser.add_argument('--pho_scale', type=float, default=1.)
parser.add_argument('--grad_scale', type=float, default=1.)
parser.add_argument('--reg_scale', type=float, default=1.)
parser.add_argument('--tv_scale', type=float, default=0.1)
parser.add_argument('--sr_scale', type=int, default=1)  # downsample ratio before computing loss

# DPS configs
parser.add_argument('--dps_steps', type=int, default=1000)
parser.add_argument('--eta', type=float, default=1.)
parser.add_argument('--n_dpps', type=int, default=20)
parser.add_argument('--vis_freq', type=int, default=100)

# lighting configs
parser.add_argument('--shade_init', type=int, default=1)
parser.add_argument('--sche_light', type=int, default=1)
parser.add_argument('--main_light_type', type=str, default="splitsum")
parser.add_argument('--opt_light_iter', type=int, default=300)
parser.add_argument('--light_res', type=int, default=96)  # resolution of the light map
parser.add_argument('--baseline', type=float, default=0)  # if True, do not use TGL
parser.add_argument('--env_light_path', type=str, default="xxx")

# reflectance initlization configs
parser.add_argument('--init', type=int, default=1)
parser.add_argument('--init_add_noise_step', type=int, default=600)
parser.add_argument('--init_map_root', type=str, default="data/init/init_3dscanstore")

# data configs
parser.add_argument('--data_root', type=str, default="data/20250415/old_mx_shadow1_delight/transforms.json")
parser.add_argument('--texture_path', type=str, default="workspace/build_texture/0415_old-mx-shadow1_delight/00150/uv.png")
parser.add_argument('--shadow_mask_path', type=str, default="xxx")
parser.add_argument('--log_dir', type=str, default="workspace/superpixel_debug/debug")

# model configs
parser.add_argument('--model_dir', type=str, default="models")
parser.add_argument('--uv_ds', type=int, default=4)


opt, _ = parser.parse_known_args()


import trimesh
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import kornia

import k_diffusion as K
from k_diffusion.sampling import get_ancestral_step, to_d

from utils.mesh_renderer import MeshRenderer
from utils.resizer import Resizer
from utils.embedder import get_embedder
from utils.lighting import TexelGridLighting


def total_variation(img, mask, tv_type="l1"):
    pixel_dif1 = (img[..., 1:, :] - img[..., :-1, :]) * mask[..., 1:, :]
    pixel_dif2 = (img[..., :, 1:] - img[..., :, :-1]) * mask[..., :, 1:]

    if tv_type == "l1":
        res1 = pixel_dif1.abs()
        res2 = pixel_dif2.abs()
    elif tv_type == "l2":
        res1 = pixel_dif1 ** 2
        res2 = pixel_dif2 ** 2
    else:
        raise NotImplementedError("Invalid tv_type option.")
    
    res1 = res1.sum()
    res2 = res2.sum()
    
    return res1 + res2


class CoordImage:
    def __init__(self, cfg):
        super().__init__()
        self.uv_size = 4096 // opt.uv_ds

        if cfg["type"] == "coord":
            self._make_coord_image()
        elif cfg["type"] == "posenc":
            self._make_coord_image_pos_enc(cfg["pos_enc_freq"])
        elif cfg["type"] == "noise":
            self._make_coord_image_noise()
        else:
            raise NotImplementedError

        self.coord_image = 2 * self.coord_image - 1

    def _make_coord_image(self):
        xs = torch.arange(self.uv_size)[None, ...].repeat(self.uv_size, 1) / self.uv_size
        ys = torch.arange(self.uv_size)[..., None].repeat(1, self.uv_size) / self.uv_size
        self.coord_image = torch.stack([xs, ys], dim=0)  # [c,uv_size,uv_size]
        # self.coord_image = 2 * self.coord_image - 1  # to [-1,1]

    def _make_coord_image_pos_enc(self, pos_enc_freq):
        xs = torch.arange(self.uv_size)[None, ...].repeat(self.uv_size, 1) / self.uv_size
        ys = torch.arange(self.uv_size)[..., None].repeat(1, self.uv_size) / self.uv_size
        coord_image = torch.stack([xs, ys], dim=-1)  # [uv_size,uv_size,c]
        coord_image = 2 * coord_image - 1  # to [-1,1]
        embedder, out_dim = get_embedder(
            multires=pos_enc_freq, input_dims=2,
        )
        coord_image = embedder(coord_image)
        coord_image = (coord_image + 1) / 2  # to [0,1]
        self.coord_image = coord_image.permute(2, 0, 1).contiguous()
    
    def _make_coord_image_noise(self):
        self.coord_image = torch.randn(3, self.uv_size, self.uv_size)
        self.coord_image = (self.coord_image + 1) / 2


def compute_init_shading(diff, shade, mask):
    eps = 1e-8
    shading = shade / (diff + eps)
    light_color = torch.sum(shading * mask, dim=(-1, -2)) / torch.sum(mask, dim=(-1, -2))
    return light_color[0]  # [3]


def degrade_operator(x):
    ks = 3
    sigma = 0.8
    return kornia.filters.gaussian_blur2d(
        x, kernel_size=(ks, ks), sigma=(sigma, sigma)
    )


def compute_skin_tone(img, mask):
    skin_tone = torch.sum(img * mask, dim=(-1, -2)) / torch.sum(mask, dim=(-1, -2))
    return skin_tone


class DiffusionSampler:
    def __init__(self, opt):
        x1 = 128
        x2 = 896
        y1 = 64
        y2 = 832
        
        self.log_dir = opt.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.device = "cuda"
        self.pho_scale = opt.pho_scale  # loss weight for photometric guidance
        self.dps_steps = opt.dps_steps
        self.use_add_noise_init = opt.init == 1
        self.init_add_noise_step = opt.init_add_noise_step
        self.dps_eta = opt.eta
        self.use_sche_light = opt.sche_light
        self.use_shade_init = opt.shade_init == 1
        self.use_baseline = opt.baseline == 1
        self.gamma = 2.2
        self.sr_degrade = None

        # load diffusion model
        self.model, model_config, self.coord_image = self._load_model(opt.model_dir)
        self.coord_image = self.coord_image.to(self.device)
        self.sigma_min = model_config['sigma_min']
        self.sigma_max = model_config['sigma_max']

        # load geometry
        self.mesh_renderer = MeshRenderer(self.device)
        self.uv_reso_h, self.uv_reso_w = 4096 // opt.uv_ds, 4096 // opt.uv_ds
        data_root = opt.data_root
        self._load_geometry(os.path.join(data_root, "final_hack.obj"))
        
        # load reflectance map for initialization
        self.diff_albedo_lr = self._load_img(os.path.join(opt.init_map_root, "diff.png")) ** self.gamma
        self.spec_albedo_lr = self._load_img(os.path.join(opt.init_map_root, "spec.png"))
        self.tangent_normal_lr = self._load_img(os.path.join(opt.init_map_root, "tan_normal.png"))
        self.tangent_normal_lr = 2 * self.tangent_normal_lr - 1
        self.tangent_normal_lr = F.normalize(self.tangent_normal_lr, dim=1)
        self.scale_factor = self.uv_reso_w / self.diff_albedo_lr.shape[-1]  # super-resolution rate
        cur_resizer = Resizer(
            in_shape=self.diff_albedo_lr.shape, scale_factor=self.scale_factor,
        ).to(self.device)
        self.diff_albedo_lr_4k = cur_resizer(self.diff_albedo_lr.to(self.device))  # [1,3,uh,uw]
        self.spec_albedo_lr_4k = cur_resizer(self.spec_albedo_lr.to(self.device))[:, :1]  # [1,1,uh,uw]
        self.tangent_normal_lr_4k = cur_resizer(self.tangent_normal_lr.to(self.device))  # [1,3,uh,uw]

        # load texture map
        self.texture_4k = self._load_img(opt.texture_path).to(self.device) ** self.gamma        
        self.texture_4k = degrade_operator(self.texture_4k)
        self.face_mask = self._load_img("assets/face_mask.png").to(self.device)[:, :1]
        
        torch.cuda.empty_cache()

        # create shadow mask
        shaodw_prior = self._load_img("assets/shadow_prior_mask.png").to(self.device)[:, :1]            
        shadow_mask = self._load_img(opt.shadow_mask_path).to(self.device)[0]
        shadow_mask = (shadow_mask[0] == 1).float() * \
                (shadow_mask[1] == 1).float() * (shadow_mask[2] == 1).float()
        shadow_mask = shadow_mask * shaodw_prior
        shadow_mask = shadow_mask[..., y1:y2, x1:x2]
        self.main_light_mask = 1 - shadow_mask[0, 0]
        edge_ks = 5
        self.tv_loss_mask = self.main_light_mask[None, None, ...] - kornia.morphology.erosion(
            self.main_light_mask[None, None, ...],
            kernel=torch.ones(edge_ks, edge_ks).to(self.device),
        )
        save_image(self.tv_loss_mask, os.path.join(self.log_dir, "tv_loss_mask.png"))
        save_image(shadow_mask, os.path.join(self.log_dir, "shadow_mask.png"))

        # initialize lighting to resolve albedo-lighting ambiguity
        skin_color_mask = self._load_img("assets/skin_color_mask.png").to(self.device)[:, :1]
        init_shade = compute_init_shading(
            diff=self.diff_albedo_lr_4k, shade=self.texture_4k, mask=skin_color_mask,
        )
        self.skin_color_mask = skin_color_mask

        # create Texel-Grid Lighting model
        self.envlight = TexelGridLighting(
            device=self.device,
            seg=(shadow_mask + self.tv_loss_mask > 0).float(),
            init_shade=init_shade,
            use_init_shade=self.use_shade_init,
            main_light_type=opt.main_light_type,
            operator=degrade_operator,
            envlight_path=opt.env_light_path,
            resolution=opt.light_res,
        )

        # save commandline arguments
        args_dict = vars(opt)
        with open(os.path.join(self.log_dir, "opt.txt"), "w") as f:
            f.write("python run_wildcap.py \\\n")
            for k in args_dict.keys():
                f.write(f"    --{k} {args_dict[k]} \\\n")

    def _load_model(self, model_dir):
        config_path = os.path.join(model_dir, "config.json")
        config = K.config.load_config(config_path)
        
        ci = CoordImage(config['cond'])
        coord_image = ci.coord_image
        coord_image = coord_image[None, ...]  # [1,2,uh,uw] in cpu
        config['cond']['cond_channels'] = ci.coord_image.shape[0]

        model_config = config['model']

        ckpt_path = os.path.join(model_dir, "ckpt.pth")
        inner_model = K.config.make_model(config).eval().requires_grad_(False).to(self.device)
        inner_model.load_state_dict(torch.load(ckpt_path)["model_ema"])
        model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'], data_channel=model_config['input_channels'])

        return model, model_config, coord_image

    def _load_geometry(self, mesh_uv_path):
        device = self.device
        mesh = trimesh.load_mesh(mesh_uv_path)
        uv = torch.from_numpy(mesh.visual.uv).to(device).float()  # [v,2]
        normal = torch.from_numpy(mesh.vertex_normals).to(device).float()  # [v,3]
        normal = F.normalize(normal, dim=-1)
        faces = torch.from_numpy(mesh.faces).to(device)  # [f,3]
        self.mesh = mesh
        self.uv = uv[None, ...]  # [1,v,2]
        self.uv = 2 * self.uv - 1
        self.uv[..., 1] *= -1
        self.faces = faces[None, ...]  # [1,v,3]

        attrs = normal[None, ...]

        # render the coarse normal map in UV space for comput shading
        uv_mesh_dict = {
            "faces": self.faces,
            "vertice": torch.cat([self.uv, torch.ones_like(self.uv[..., :1])], dim=-1),  # [1,v,3]
            "attributes": attrs,  # [1,v,3]
            "size": (self.uv_reso_h, self.uv_reso_w),
        }
        attrs_uv, uv_pix_to_face = self.mesh_renderer.render_ndc(uv_mesh_dict)  # [1,6,uv,uv]
        
        normal_uv = attrs_uv[:, :3]
        self.object_normal_lr_4k = torch.clone(normal_uv)  # [1,3,uh,uw]

    def _load_img(self, pth):
        return transforms.ToTensor()(Image.open(pth))[None, :3, ...].cpu()

    def sampling(self):
        '''
        only process a small number of patch to acclerate debug
        '''
        log_dir = self.log_dir
        tb_record = False
        if tb_record:
            writer = SummaryWriter(log_dir)

        x1 = 128
        x2 = 896
        y1 = 64
        y2 = 832

        num_chns = 3 + 3 + 1

        sigmas = K.sampling.get_sigmas_karras(self.dps_steps, self.sigma_min, self.sigma_max, rho=7., device=self.device)
        ADD_NOISE_STEPS = self.init_add_noise_step
        if self.use_add_noise_init:
            sigma_max = sigmas[-ADD_NOISE_STEPS]
            sigmas = sigmas[-ADD_NOISE_STEPS:]
        else:
            sigma_max = self.sigma_max
        
        if True:  # set initial noise
            noise = torch.randn(
                1, 
                num_chns,
                y2 - y1,
                x2 - x1,
            ).to(self.device)

            if self.use_add_noise_init:
                x_init = torch.cat([
                    2 * self.diff_albedo_lr_4k - 1,
                    self.tangent_normal_lr_4k,
                    2 * self.spec_albedo_lr_4k - 1,
                ], dim=1)[..., y1:y2, x1:x2]
                xt = x_init + noise * sigma_max
            else:
                xt = noise * sigma_max

        if True:  # set lighting optimizer
            params_main_light = [
                {"params": self.envlight.main_light.parameters(), "lr": 0.01},
            ]
            params_sp_light = [
                {"params": self.envlight.sp_light, "lr": 0.01},
            ]
            optimizer_main_light = torch.optim.Adam(params_main_light)
            scheduler_main_light = torch.optim.lr_scheduler.ExponentialLR(optimizer_main_light, gamma=0.995)
            optimizer_sp_light = torch.optim.Adam(params_sp_light)
            scheduler_sp_light = torch.optim.lr_scheduler.ExponentialLR(optimizer_sp_light, gamma=0.995)

        for step_idx in tqdm(range(len(sigmas) - 1)):
            if True:  # posterior sampling step
                xt.detach_()
                i = step_idx
                extra_args = {}
                s_in = xt.new_ones([xt.shape[0]])
                xt.requires_grad_(True)
                x_input = torch.cat([xt, self.coord_image[..., y1:y2, x1:x2]], dim=1)
                denoised = self.model(x_input, sigmas[i] * s_in, **extra_args)
                sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=self.dps_eta)
                d = to_d(xt, sigmas[i], denoised)

                cur_diff = (denoised[:, :3] + 1) / 2
                cur_spec = (denoised[:, 6:7] + 1) / 2
                cur_normal = denoised[:, 3:6]
                cur_normal = F.normalize(cur_normal, dim=1)
                shading, _, _ = self.envlight(
                    face_texture=cur_diff, 
                    face_norm=self.object_normal_lr_4k[..., y1:y2, x1:x2], 
                    detach=True,
                    baseline=self.use_baseline,
                )
                
                grad_loss_mask = 0.
                pho_loss_mask = 1.
                face_mask = self.face_mask[..., y1:y2, x1:x2]

                if self.sr_degrade is None:
                    self.sr_degrade = Resizer(
                        in_shape=face_mask.shape, scale_factor=1./opt.sr_scale,
                    ).to(self.device)
                
                loss_color = F.mse_loss(
                    self.sr_degrade(shading * face_mask * pho_loss_mask),
                    self.sr_degrade(torch.clone(self.texture_4k[..., y1:y2, x1:x2]) * face_mask * pho_loss_mask),
                    reduction="sum"
                )

                gt_grad = kornia.filters.spatial_gradient(self.sr_degrade(torch.clone(self.texture_4k[..., y1:y2, x1:x2])))
                shading_grad = kornia.filters.spatial_gradient(self.sr_degrade(shading))
                loss_grad = F.mse_loss(
                    shading_grad * self.sr_degrade(face_mask * grad_loss_mask),
                    gt_grad * self.sr_degrade(face_mask * grad_loss_mask),
                    reduction="sum"
                )
                
                loss = loss_color + loss_grad * opt.grad_scale
                loss.backward()

                if tb_record:
                    writer.add_scalar("loss_total", loss_color.item(), step_idx)
                
                norm_grad = torch.clone(xt.grad) * face_mask

                while torch.abs(norm_grad).max().item() > 0.1:
                    norm_grad = norm_grad * 0.1

                shading_light, shade, shade_main = self.envlight(
                    face_texture=cur_diff.detach(), 
                    face_norm=self.object_normal_lr_4k[..., y1:y2, x1:x2].detach(), 
                    detach=False,
                    baseline=self.use_baseline,
                )
                loss_reg = (F.relu(shade - shade_main) ** 2).sum()
                loss_light = F.mse_loss(
                    self.sr_degrade(shading_light * face_mask * pho_loss_mask),
                    self.sr_degrade(torch.clone(self.texture_4k[..., y1:y2, x1:x2]) * face_mask * pho_loss_mask),
                    reduction="sum"
                )
                                
                loss_tv = total_variation(
                    shade, self.tv_loss_mask,
                )
                loss_tv = loss_tv.mean()
                loss_light_total = loss_light + loss_reg * opt.reg_scale + loss_tv * opt.tv_scale

                if step_idx <= 100:
                    optimizer = optimizer_main_light
                    scheduler = scheduler_main_light
                else:
                    optimizer = optimizer_sp_light
                    scheduler = scheduler_sp_light  

                optimizer.zero_grad()
                loss_light_total.backward()
                if step_idx <= opt.opt_light_iter:
                    optimizer.step()
                    if self.use_sche_light:
                        scheduler.step()
                
                # we use DDPS, an improved version of DPS
                # see more details in the paper "Diffusion Posterior Proximal Sampling for Image Restoration"
                dt = sigma_down - sigmas[i]
                xt.detach_()
                x_new = xt + d * dt - norm_grad * self.pho_scale
                with torch.no_grad():
                    if sigmas[i + 1] > 0:
                        # compute Ax_{t-1}^{*}
                        # x_{t-1} = x_{t} + dt * (x_{t} - x{0}) / sigma[i]
                        # x_{t-1} = (1 + dt / sigma[i]) * x_{t} - dt / sigma[i] * x_{0}
                        xt_diff = (xt[:, :3] + 1) / 2
                        A_xt, lighting, _ = self.envlight(
                            face_texture=xt_diff,
                            face_norm=self.object_normal_lr_4k[..., y1:y2, x1:x2], 
                            baseline=self.use_baseline,
                        )
                        c1 = 1 + dt / sigmas[i]
                        c2 = dt / sigmas[i]
                        # my_d = (xt - denoised) / sigmas[i]
                        # diff = (my_d - d).abs()
                        # print(torch.max(diff))
                        A_x0_star = self.texture_4k[..., y1:y2, x1:x2]
                        A_xt_1_star = c1 * A_xt - c2 * A_x0_star

                        _, c, h, w = xt.shape
                        cur_noise = torch.randn(opt.n_dpps, c, h, w, device=self.device)
                        xt_1 = x_new + cur_noise * sigma_up
                        xt_1_diff = (xt_1[:, :3] + 1) / 2
                        A_xt_1 = lighting * xt_1_diff
                        cur_loss = (A_xt_1 * face_mask - A_xt_1_star * face_mask).pow(2).sum(dim=(-1, -2, -3))
                        best_idx = torch.argmin(cur_loss)
                        x_new = torch.clone(xt_1[best_idx:best_idx + 1])

                xt = x_new

                if step_idx % opt.vis_freq == 0 or step_idx == len(sigmas) - 2:
                    cur_log_dir = os.path.join(log_dir, "%05d" % step_idx)
                    self.save_maps(
                        cur_log_dir=cur_log_dir,
                        diff=cur_diff, spec=cur_spec, normal=cur_normal, shading=shading,
                    )
                    save_image(shade / torch.max(shade), os.path.join(cur_log_dir, "shade.png"))
                    save_image(shade_main / torch.max(shade), os.path.join(cur_log_dir, "shade_main.png"))
                    torch.save(self.envlight.state_dict(), os.path.join(cur_log_dir, "envmap.pth"))

    def save_maps(self, cur_log_dir, diff, spec, normal, shading):
        x1 = 128
        x2 = 896
        y1 = 64
        y2 = 832
        os.makedirs(cur_log_dir, exist_ok=True)

        res_diff = torch.zeros(1, 3, 1024, 1024).to(self.device)
        res_diff[..., y1:y2, x1:x2] = diff ** (1 / self.gamma)

        torch.nan_to_num_(res_diff, nan=0.0, posinf=0.0, neginf=0.0)

        tgt_diff_skin_tone = compute_skin_tone(
            img=self.diff_albedo_lr_4k**(1/self.gamma), mask=self.skin_color_mask,
        )
        cur_diff_skin_tone = compute_skin_tone(
            img=res_diff, mask=self.skin_color_mask,
        )
        res_diff = res_diff * (tgt_diff_skin_tone / cur_diff_skin_tone)[..., None, None]

        res_spec = torch.zeros(1, 1, 1024, 1024).to(self.device)
        res_spec[..., y1:y2, x1:x2] = spec

        res_normal = torch.zeros(1, 3, 1024, 1024).to(self.device)
        res_normal[..., y1:y2, x1:x2] = (normal + 1) / 2

        res_shading = torch.zeros(1, 3, 1024, 1024).to(self.device)
        res_shading[..., y1:y2, x1:x2] = shading ** (1 / self.gamma)

        save_image(res_diff, os.path.join(cur_log_dir, "diff_sr.png"))
        save_image(res_spec, os.path.join(cur_log_dir, "spec_sr.png"))
        save_image(res_normal, os.path.join(cur_log_dir, "normal_sr.png"))
        save_image(res_shading, os.path.join(cur_log_dir, "shading_sr.png"))

    
if __name__ == "__main__":
    ds = DiffusionSampler(opt)
    ds.sampling()
