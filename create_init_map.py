import os

from PIL import Image
import torch
from torchvision.utils import save_image
from torchvision import transforms

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--skin_tone', type=str, default="[0.6902, 0.5373, 0.4471]")
parser.add_argument('--save_root', type=str, default="workspace/ljf/sample_dataset")


opt, _ = parser.parse_known_args()


def load_img(pth):
    return transforms.ToTensor()(Image.open(pth))


skin_color_mask = load_img("assets/skin_color_mask.png")
spec_template = load_img("assets/init_map/spec_template.png")
normal_template = load_img("assets/init_map/normal_template.png")

spec_template = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC)(spec_template)
normal_template = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC)(normal_template)

save_root = opt.save_root
os.makedirs(save_root, exist_ok=True)

skin_tone = torch.tensor(eval(opt.skin_tone))
diff_map_uniform = skin_tone[..., None, None] * torch.ones_like(skin_color_mask)

save_image(diff_map_uniform, os.path.join(save_root, "diff.png"))
save_image(spec_template, os.path.join(save_root, "spec.png"))
save_image(normal_template, os.path.join(save_root, "tan_normal.png"))
