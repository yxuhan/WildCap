import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import torch.nn as nn
import torch.nn.functional as F

from splitsum import util
from splitsum import renderutils as ru
from utils.resizer import Resizer


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


class TexelGridLighting(nn.Module):
    def __init__(
        self, 
        device,
        seg,
        main_light_type="splitsum",
        init_shade=None,
        use_init_shade=False,
        operator=None,
        envlight_path=None,
        resolution=192,
    ):
        super().__init__()
        self.SH = SH()
        self.device = device
        self.init_shade = init_shade
        self.use_init_shade = use_init_shade
        self.default_init_ss = 0.75
        self.main_light_type = main_light_type
        self.ss_sh_scale = self.compute_sh_ss_scale()
        self.operator = operator
        self.envlight_path = envlight_path
        self.resolution = resolution

        self.sp_light_mask = seg  # [h,w]
        
        self.set_sp_light()
        self.set_main_light()

    def set_main_light(self):
        if self.main_light_type == "splitsum":
            if not os.path.exists(self.envlight_path):
                init_color = self.compute_init_ss_color()
                base = torch.ones(6, 512, 512, 3, dtype=torch.float32, device=self.device) * init_color
                self.main_light = SplitSum(
                    base, requires_grad=True,
                )
            else:
                base = torch.load(self.envlight_path, map_location=self.device)["main_light.base"]
                self.main_light = SplitSum(
                    base, requires_grad=False,
                )
        elif self.main_light_type == "sh":
            pass
        else:
            raise NotImplementedError

    def set_sp_light(self):
        gamma = torch.zeros(
            3, 9, self.resolution, self.resolution
        ).to(self.device)
        self.sp_light = torch.nn.parameter.Parameter(
            gamma, requires_grad=True,
        )

        self.sp_light_resizer = Resizer(
            in_shape=self.sp_light.shape, 
            scale_factor=self.sp_light_mask.shape[-1]/self.resolution,
        ).to(self.device)
    
    def make_sp_light_map(self):
        res = self.sp_light
        res = self.sp_light_resizer(res)  # [3,9,h,w]
        res = res.permute(0, 2, 3, 1).contiguous()  # [3,h,w,9]
        return res

    def compute_sh_ss_scale(self):
        with torch.no_grad():
            # ss lighting
            base = torch.ones(6, 512, 512, 3, dtype=torch.float32, device=self.device)
            sslight = SplitSum(
                base, requires_grad=False,
            )
            sslight.build_mips()
            normal_map = torch.randn(1, 3, 256, 256).to(self.device)
            normal_map = F.normalize(normal_map, dim=0)
            res_ss = sslight.shade_diffuse(normal_map)
            res_ss = torch.mean(res_ss)
            # sh lighting
            gamma_map = torch.zeros(9).to(self.device)
            gamma_map[0] = 1.
            res_sh = self.compute_sh_shading(normal_map, gamma_map)
            res_sh = torch.mean(res_sh)
        return res_ss / res_sh

    def compute_init_sh_color(self):
        if self.use_init_shade:
            a, c = self.SH.a, self.SH.c
            init_shading = self.init_shade / a[0] / c[0]
            return init_shading
        else:
            return self.default_init_ss * self.ss_sh_scale

    def compute_init_ss_color(self):
        if self.use_init_shade:
            init_shading_sh = self.compute_init_sh_color()
            return init_shading_sh / self.ss_sh_scale
        else:
            return self.default_init_ss

    def compute_sh_shading(self, normal, gamma_map=1.):
        face_norm = normal.permute(0, 2, 3, 1)
        a, c = self.SH.a, self.SH.c
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

        shading = torch.sum(Y * gamma_map, dim=-1)
        return shading

    def forward(self, face_texture, face_norm, detach=False, baseline=False):
        diffuse = face_texture
        normal = face_norm
        
        sp_gamma_map = self.make_sp_light_map()  # [3,h,w,9]
        sp_shading = self.compute_sh_shading(normal, sp_gamma_map)
        
        if self.main_light_type == "splitsum":
            self.main_light.build_mips()
            ss_shading = self.main_light.shade_diffuse(normal)
            if baseline:
                shading = ss_shading
            else:
                # shading = ss_shading * self.main_light_mask + (1 - self.main_light_mask) * sp_shading
                shading = ss_shading + sp_shading * self.sp_light_mask
            with torch.no_grad():
                main_shading_mask = (ss_shading[:, 0:1] == 0).float() * (ss_shading[:, 1:2] == 0).float() * (ss_shading[:, 2:3] == 0).float()
                main_shading_mask = 1 - main_shading_mask
            shading = shading * main_shading_mask
            main_shading = ss_shading * main_shading_mask
        elif self.main_light_type == "sh":
            main_gamma_map = self.sp_light[0]  # [3,1,1,9]
            shading = sp_shading
            main_shading = self.compute_sh_shading(normal, main_gamma_map)
        else:
            raise NotImplementedError

        main_shading = main_shading.detach()
        if detach:
            shading = shading.detach()
        face_color = diffuse * shading
        if self.operator is not None:
            face_color = self.operator(face_color)
        return face_color, shading, main_shading

    def forward_main_light(self, face_texture, face_norm, detach=False):
        diffuse = face_texture
        normal = face_norm
        
        if self.main_light_type == "splitsum":
            self.main_light.build_mips()
            shading = self.main_light.shade_diffuse(normal)
        else:
            raise NotImplementedError

        if detach:
            shading = shading.detach()
        face_color = diffuse * shading
        return face_color


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


class SplitSum(nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base, requires_grad=False):
        super(SplitSum, self).__init__()
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=requires_grad)
        self.register_parameter('env_base', self.base)

    def clone(self):
        return SplitSum(self.base.clone().detach())

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
    
    def shade_diffuse(self, normal):
        '''
        input:
            diff: [1,3,h,w]
        output:
            shading: [1,3,h,w]
        '''
        nview = normal.shape[0]
        npatch = 1
        h, w = normal.shape[-2:]

        normal_pad = normal[None, ...].repeat(nview, 1, 1, 1, 1)
        
        normal_vector = normal_pad.permute(0, 1, 3, 4, 2)
        normal_vector = normal_vector.reshape(nview * npatch, h, w, -1)
        
        # compute diffuse shading
        diff_shading = dr.texture(
            self.diffuse[None, ...], normal_vector.contiguous(), filter_mode='linear', boundary_mode='cube'
        )
        diff_shading = diff_shading.reshape(nview, npatch, h, w, -1)
        diff_shading = diff_shading.permute(0, 1, 4, 2, 3)  # [nview,npatch,3,h,w]

        return diff_shading[0]    


if __name__ == "__main__":
    pass
