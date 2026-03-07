'''
main light用split sum建模
其他每个superpixel的light用SH建模
'''


import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import kornia
import torch.nn as nn
from torchvision.utils import save_image

from . import util
from . import renderutils as ru


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


class SHLighting(nn.Module):
    def __init__(
        self, 
        device, 
        seg,
        ks,
        face_mask,
        init_shade,
        use_init_shade=False,
        use_splitsum=False,
    ):
        super().__init__()
        self.SH = SH()
        self.device = device

        seg = seg - 1  # so that index start from 0
        label_list = np.unique(seg)
        self.seg = torch.from_numpy(seg).to(self.device)  # [h,w]
        self.num_light = len(label_list)
        self.main_light_mask = (self.seg == 0).float()
        self.main_light_mask = ((self.main_light_mask + 1 - face_mask) > 0).float()

        # init light color
        if use_init_shade:
            base = torch.ones(6, 512, 512, 3, dtype=torch.float32, device='cuda')
            tmp_light = EnvironmentLight(
                base, requires_grad=False,
            )
            tmp_light.build_mips()
            tmp_shading = torch.mean(tmp_light.diffuse)
            light_color = init_shade / tmp_shading
            base = torch.ones(6, 512, 512, 3, dtype=torch.float32, device='cuda') * light_color
            self.sslight = EnvironmentLight(
                base, requires_grad=True,
            )

            a, c = self.SH.a, self.SH.c
            init_shading = init_shade / a[0] / c[0]
            gamma = torch.zeros(self.num_light, 3, 1, 1, 9).to(self.device)
            gamma[..., 0] += init_shading[..., None, None]
            self.gamma = torch.nn.parameter.Parameter(
                gamma, requires_grad=True,
            )
        else:
            base = torch.ones(6, 512, 512, 3, dtype=torch.float32, device='cuda') * 0.75
            self.sslight = EnvironmentLight(
                base, requires_grad=True,
            )
            self.sslight.build_mips()
            ss_shading = torch.mean(self.sslight.diffuse)
            a, c = self.SH.a, self.SH.c
            init_shading = ss_shading / a[0] / c[0]

            gamma = torch.zeros(self.num_light, 3, 1, 1, 9).to(self.device)
            gamma[..., 0] += init_shading
            self.gamma = torch.nn.parameter.Parameter(
                gamma, requires_grad=True,
            )

        self.ks = ks
        self.use_splitsum = use_splitsum
        if self.use_splitsum:
            self.main_light_mask = kornia.filters.gaussian_blur2d(
                self.main_light_mask, kernel_size=ks, sigma=(ks//3, ks//3),
            )
    
    def make_light_map(self):
        gamma = self.gamma
        res2 = gamma[self.seg]  # [h,w,3,1,1,9]
        res2 = res2[:, :, :, 0, 0, :]  # [h,w,3,9]
        res2 = res2.permute(2, 3, 0, 1).contiguous()  # [3,9,h,w]
        if self.ks > 0:
            ks = self.ks
            res2 = kornia.filters.gaussian_blur2d(res2, kernel_size=ks, sigma=(ks/3, ks/3))
            res2 = res2.permute(0, 2, 3, 1).contiguous()
            # res2 = res2[:, :, :, 0, 0, :]  # [h,w,3,9]
            # res2 = res2.permute(2, 0, 1, 3).contiguous()  # [3,h,w,9]
        return res2
    
    def forward(self, face_texture, face_norm, detach=False):
        """
        face_texture: [1,3,h,w]
        gamma: [3,9]
        face_norm: [1,3,h,w]
        """
        # self.sslight.build_mips()
        # face_color_ss, main_shading = self.sslight.shade_hyx_diffuse(face_texture, face_norm)

        gamma_map = self.make_light_map()  # [3,h,w,9]
        with torch.no_grad():
            main_gamma_map = self.gamma[0]

        a, c = self.SH.a, self.SH.c
        face_norm = face_norm.permute(0, 2, 3, 1)  # [1,h,w,3]

        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)  # [1,h,w,9]

        # r = torch.sum(gamma[0] * Y, dim=-1)  # [1,h,w]
        # g = torch.sum(gamma[1] * Y, dim=-1)  # [1,h,w]
        # b = torch.sum(gamma[2] * Y, dim=-1)  # [1,h,w]
        shading = torch.sum(Y * gamma_map, dim=-1)
        if self.use_splitsum:
            return self.forward_splitsum(
                face_texture=face_texture,
                face_norm=face_norm.permute(0, 3, 1, 2),
                shading_sh=shading,
                detach=detach,
            )
        main_shading = torch.sum(Y * main_gamma_map, dim=-1)
        # with torch.no_grad():
        #     main_shading = face_color_ss / (face_texture + 1e-8)

        # print(torch.max(shading), torch.min(shading))
        # mask = (shading < 0).float()
        # from torchvision.utils import save_image
        # save_image(mask, "neg_mask.jpg")
        if detach:
            shading = shading.detach()
        face_color_sh = shading * face_texture

        # face_color = self.main_light_mask * face_color_ss + (1 - self.main_light_mask) * face_color_sh
        # with torch.no_grad():
        #     main_shading_mask = (main_shading[:, 0:1] == 0).float() * (main_shading[:, 1:2] == 0).float() * (main_shading[:, 2:3] == 0).float()
        #     main_shading_mask = 1 - main_shading_mask
        
        return face_color_sh, shading, main_shading.detach()

    def forward_splitsum(self, face_texture, face_norm, shading_sh, detach=False):
        self.sslight.build_mips()
        face_color_ss, main_shading = self.sslight.shade_hyx_diffuse(face_texture, face_norm)
        with torch.no_grad():
            main_shading_mask = (main_shading[:, 0:1] == 0).float() * (main_shading[:, 1:2] == 0).float() * (main_shading[:, 2:3] == 0).float()
            main_shading_mask = 1 - main_shading_mask
        
        shading = self.main_light_mask * main_shading + (1 - self.main_light_mask) * shading_sh
        if detach:
            shading = shading.detach()
        
        face_color = shading * face_texture
        return face_color, shading, main_shading.detach()


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out


class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base, requires_grad=False):
        super(EnvironmentLight, self).__init__()
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=requires_grad)
        self.register_parameter('env_base', self.base)

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)
    
    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))
    
    def shade_hyx_diffuse(self, diff, normal):
        '''
        和co-located shading对齐接口
        '''
        nview = diff.shape[0]
        npatch = 1
        h, w = diff.shape[-2:]

        diff_pad = diff[None, ...].repeat(nview, 1, 1, 1, 1)
        normal_pad = normal[None, ...].repeat(nview, 1, 1, 1, 1)
        
        diff_vector = diff_pad.permute(0, 1, 3, 4, 2)  # [view,b,h,w,c]
        normal_vector = normal_pad.permute(0, 1, 3, 4, 2)

        normal_vector = normal_vector.reshape(nview * npatch, h, w, -1)
        
        # compute diffuse shading
        diff_shading = dr.texture(
            self.diffuse[None, ...], normal_vector.contiguous(), filter_mode='linear', boundary_mode='cube'
        )
        diff_shading = diff_shading.reshape(nview, npatch, h, w, -1)
        diff_color = diff_shading * diff_vector
        diff_color = diff_color.permute(0, 1, 4, 2, 3)  # [nview,npatch,3,h,w]
        diff_shading = diff_shading.permute(0, 1, 4, 2, 3)  # [nview,npatch,3,h,w]

        return diff_color[0], diff_shading[0]
    

def _load_env_hdr(fn, scale=1.0):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l


def load_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr(fn, scale)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]


def hyx_load_env_hdr(fn, scale=1.0, requires_grad=False):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap, requires_grad=requires_grad)
    l.build_mips()

    return l


def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    util.save_image_raw(fn, color.detach().cpu().numpy())


def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base)

