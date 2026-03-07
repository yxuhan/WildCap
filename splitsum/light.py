import os
import numpy as np
import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru


def project_points_onto_sphere(a, r, x, d):
    """
    将点沿着方向向量投影到球面上
    
    参数:
    a (torch.Tensor): 球心坐标，形状为 [b, 3]
    r (torch.Tensor): 球半径，形状为 [b, 1]
    x (torch.Tensor): 点的位置，形状为 [b, n, 3]
    d (torch.Tensor): 方向向量，形状为 [b, n, 3]
    
    返回:
    torch.Tensor: 投影点的坐标，形状为 [b, n, 3]
    """
    # 确保输入张量维度正确
    assert a.dim() == 2 and a.size(1) == 3, "球心 a 必须是 [b, 3] 张量"
    assert r.dim() == 2 and r.size(1) == 1, "半径 r 必须是 [b, 1] 张量"
    assert x.dim() == 3 and x.size(2) == 3, "点 x 必须是 [b, n, 3] 张量"
    assert d.dim() == 3 and d.size(2) == 3, "方向 d 必须是 [b, n, 3] 张量"
    
    # 规范化方向向量
    d_normalized = torch.nn.functional.normalize(d, dim=2)
    
    # 计算从球心到点的向量
    oc = x - a.unsqueeze(1)  # [b, n, 3]
    
    # 计算二次方程的系数: t^2 + 2*dot(oc, d)*t + (dot(oc, oc) - r^2) = 0
    a_coeff = 1.0
    b_coeff = 2.0 * torch.sum(oc * d_normalized, dim=2, keepdim=True)  # [b, n, 1]
    c_coeff = torch.sum(oc * oc, dim=2, keepdim=True) - r.pow(2).unsqueeze(1)  # [b, n, 1]
    
    # 计算判别式
    discriminant = b_coeff.pow(2) - 4 * a_coeff * c_coeff  # [b, n, 1]
    
    # 计算两个解
    sqrt_discriminant = torch.sqrt(discriminant.clamp(min=0))
    t = (-b_coeff + sqrt_discriminant) / (2 * a_coeff)  # [b, n, 1]
    
    # 计算投影点
    projection = x + t * d_normalized
    
    return projection


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

    def shade_hyx_diffuse_near(self, diff, normal, verts, mask):
        # compute local light probe center and radius
        indices_valid = torch.nonzero(mask, as_tuple=True)
        verts_valid = verts[:, :, indices_valid[0], indices_valid[1]]  # [b,3,n]
        center = torch.mean(verts_valid, dim=-1, keepdim=True)  # [b,3,1]
        dist = torch.sum((verts_valid - center) ** 2, dim=1, keepdim=True).sqrt()  # [b,1,n]
        radius, _ = torch.max(dist, dim=-1)  # [b,1]
        center = center[..., 0]  # [b,3]
        
        # project vertives to the sphere
        # (verts_valid + t * n - center) ** 2 = radius ** 2
        normal_valid = normal[:, :, indices_valid[0], indices_valid[1]]
        normal_valid = normal_valid.permute(0, 2, 1)  # [b,n,3]
        verts_valid = verts_valid.permute(0, 2, 1)  # [b,n,3]
        verts_proj = project_points_onto_sphere(
            a=center, r=radius, x=verts_valid, d=normal_valid,
        )  # [b,n,3]
        new_normal = torch.nn.functional.normalize(verts_proj - center[:, None, :], dim=-1)
        new_normal = new_normal.permute(0, 2, 1)  # [b,3,n]
        # from torchvision.utils import save_image
        # save_image((normal + 1) / 2, "normal_before.jpg")
        normal[:, :, indices_valid[0], indices_valid[1]] = new_normal
        # save_image((normal + 1) / 2, "normal_after.jpg")
        return self.shade_hyx_diffuse(diff, normal)
    
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
        diff_shading = diff_shading * diff_vector
        diff_shading = diff_shading.permute(0, 1, 4, 2, 3)  # [nview,npatch,3,h,w]

        return diff_shading

    def shade_hyx(self, diff, spec, normal, viewdir):
        '''
        和co-located shading对齐接口
        '''
        nview = viewdir.shape[0]
        npatch = viewdir.shape[1]
        h, w = viewdir.shape[-2:]

        diff_pad = diff[None, ...].repeat(nview, 1, 1, 1, 1)
        spec_pad = spec[None, ...].repeat(nview, 1, 1, 1, 1)
        normal_pad = normal[None, ...].repeat(nview, 1, 1, 1, 1)
        
        diff_vector = diff_pad.permute(0, 1, 3, 4, 2)  # [view,b,h,w,c]
        spec_vector = spec_pad.permute(0, 1, 3, 4, 2)
        normal_vector = normal_pad.permute(0, 1, 3, 4, 2)
        view_vector = viewdir.permute(0, 1, 3, 4, 2)

        normal_vector = normal_vector.reshape(nview * npatch, h, w, -1)
        view_vector = -view_vector.reshape(nview * npatch, h, w, -1)  # from surface to light
        ref_vector = util.reflect(view_vector, normal_vector)
        spec_vector = spec_vector.reshape(nview * npatch, h, w, -1)
        
        # compute diffuse shading
        diff_shading = dr.texture(
            self.diffuse[None, ...], normal_vector.contiguous(), filter_mode='linear', boundary_mode='cube'
        )
        diff_shading = diff_shading.reshape(nview, npatch, h, w, -1)
        diff_shading = diff_shading * diff_vector
        diff_shading = diff_shading.permute(0, 1, 4, 2, 3)  # [nview,npatch,3,h,w]

        # compute specular shading
        # Lookup FG term from lookup texture
        NdotV = torch.clamp(util.dot(view_vector, normal_vector), min=1e-4)  # [b,h,w,1]
        roughness = 0.45 * torch.ones_like(NdotV)
        fg_uv = torch.cat((NdotV, roughness), dim=-1)  # [b,h,w,2]
        if not hasattr(self, '_FG_LUT'):
            self._FG_LUT = torch.load("data/disney_brdf_lut.pkl", map_location=viewdir.device)  # [3,256,256]
            self._FG_LUT = self._FG_LUT[None, :2].permute(0, 2, 3, 1).contiguous()
            
        fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

        # Roughness adjusted specular env lookup
        miplevel = self.get_mip(roughness)
        spec = dr.texture(self.specular[0][None, ...], ref_vector.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

        # Compute aggregate lighting
        reflectance = 0.08 * fg_lookup[...,0:1] + fg_lookup[...,1:2]
        spec_shading = spec * reflectance * spec_vector
        spec_shading = spec_shading.reshape(nview, npatch, h, w, -1).permute(0, 1, 4, 2, 3)

        return diff_shading, spec_shading.expand_as(diff_shading)

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        wo = util.safe_normalize(view_pos - gb_pos)  # [b,h,w,3]

        if specular:
            roughness = ks[..., 1:2] # y component  [b,h,w,1]
            metallic  = ks[..., 2:3] # z component
            spec_col  = (1.0 - metallic)*0.04 + kd * metallic  # [b,h,w,3]
            diff_col  = kd * (1.0 - metallic)  # [b,h,w,3]
        else:
            diff_col = kd

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal  # [b,h,w,3]

        # Diffuse lookup
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        shaded_col = diffuse * diff_col

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)  # [b,h,w,1]
            fg_uv = torch.cat((NdotV, roughness), dim=-1)  # [b,h,w,2]
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
                # blue = torch.zeros_like(self._FG_LUT[..., :1])
                # vis = torch.cat([self._FG_LUT, blue], dim=-1)
                # from torchvision.utils import save_image
                # save_image(vis.permute(0, 3, 1, 2), "vis.jpg")
                
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            shaded_col += spec * reflectance

        return shaded_col * (1.0 - ks[..., 0:1]) # Modulate by hemisphere visibility


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
