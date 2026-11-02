import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from config_cropping import cfg
from CACNet import CACNet
import warnings
from torchvision import transforms
from PIL import Image, ImageOps
import random
import time

warnings.filterwarnings("ignore")

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
# Normalize the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
])


def rescale_bbox(bbox, ratio_w, ratio_h):
    bbox = np.array(bbox).reshape(-1, 4)
    bbox[:, 0] = np.floor(bbox[:, 0] * ratio_w)
    bbox[:, 1] = np.floor(bbox[:, 1] * ratio_h)
    bbox[:, 2] = np.ceil(bbox[:, 2] * ratio_w)
    bbox[:, 3] = np.ceil(bbox[:, 3] * ratio_h)
    return bbox.astype(np.float32)


def preprocess_image(image, keep_aspect_ratio):
    im_width, im_height = image.size
    if keep_aspect_ratio:
        scale = float(cfg.image_size[0]) / min(im_height, im_width)
        h = round(im_height * scale / 32.0) * 32
        w = round(im_width * scale / 32.0) * 32
    else:
        h = cfg.image_size[1]
        w = cfg.image_size[0]

    resized_image = image.resize((w, h), Image.Resampling.LANCZOS)

    if random.uniform(0, 1) > 0.5:
        resized_image = ImageOps.mirror(resized_image)

    return transform(resized_image).unsqueeze(0).to(device)


def evaluate_on_images(model, input_path: Path, output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)
    keep_aspect_ratio = False
    image_files = []
    for image_folder in input_path.iterdir():
        if not image_folder.is_dir():
            continue
        image_path = image_folder / "generated_0.jpeg"
        if image_path.exists():
            image_files.append(image_path)

    start = time.time()
    for image_file in tqdm(image_files):
        image = Image.open(image_file).convert('RGB')
        width, height = image.size
        im = preprocess_image(image, keep_aspect_ratio)
        # model inference
        logits, kcm, crop = model(im, only_classify=False)
        # convert the crop to the original image size
        crop[:, 0::2] = crop[:, 0::2] / im.shape[-1] * width
        crop[:, 1::2] = crop[:, 1::2] / im.shape[-2] * height
        pred_crop = crop.detach().cpu()
        pred_crop[:, 0::2] = torch.clip(pred_crop[:, 0::2], min=0, max=width)
        pred_crop[:, 1::2] = torch.clip(pred_crop[:, 1::2], min=0, max=height)
        best_crop = pred_crop[0].numpy().tolist()
        best_crop = [int(x) for x in best_crop]  # x1,y1,x2,y2
        # save the best crop
        cropped_img = image.crop(best_crop)
        res_path = output_dir / f"{image_file.parent.stem}.jpeg"
        cropped_img.save(res_path)
        cost_time = time.time() - start
        print('cost time: {:.2f}s, cropped image saved to {}'.format(cost_time, res_path))


if __name__ == '__main__':
    input_path = Path("./assets/input")
    output_path = Path("./assets/output")
    device = torch.device("cuda:0")
    model = CACNet(loadweights=False)
    state_dict = torch.load("./pretrained_model/best-FLMS_iou.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    evaluate_on_images(model, input_path, output_path)
