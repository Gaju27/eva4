import os
import PIL
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from images import *
from models import *
from display import *


device=has_cuda()
img_dir = 'images'
img_name = 'multiple_dogs.jpg'
img_path = os.path.join(img_dir, img_name)

pil_img = PIL.Image.open(img_path)
pil_img

torch_img = transforms.Compose([transforms.Resize((56, 56)), transforms.ToTensor()])(pil_img).to(device)
normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]



def grad_cam():
    net = ResNet18()
    configs = [dict(model_type='resnet', arch=net, layer_name='layer4')]
    
    for config in configs:
        config['arch'].to(device).eval()

    cams = [[cls.from_config(**config) for cls in (GradCAM, GradCAMpp)] for config in configs]    

    images = []
    for gradcam, gradcam_pp in cams:
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])

    return make_grid(images, nrow=5)
