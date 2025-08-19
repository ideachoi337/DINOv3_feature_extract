import os
import torch
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import os
import os.path as osp
import multiprocessing as mp
import argparse

import torchvision
from torchvision import transforms
import time

def make_transform(resize_size: int = 224):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

def get_feature(img_path, transform, model, device):
    with Image.open(img_path) as img:
        image = img.convert("RGB")
    inputs = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.cpu()

def process_subset(args, gpu_id, video_subset):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    REPO_DIR = args.repo_dir
    MODEL_DIR = args.model
    model = torch.hub.load(REPO_DIR, '_'.join(osp.basename(MODEL_DIR).split('_')[:2]), source='local', weights=MODEL_DIR).to(device).eval()
    transform = make_transform()

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    for video_name in tqdm(video_subset, desc=f"GPU {gpu_id}", position=gpu_id, leave=True):
        features = []
        frame_list = sorted(glob(osp.join(video_name, '*.jpg')))
        for frame in frame_list:
            feature = get_feature(frame, transform, model, device)
            features.append(feature)

        features = torch.cat(features, dim=0)

        save_path = osp.join(save_dir, osp.basename(video_name) + ".npy")
        np.save(save_path, features.numpy())

def main(args):
    video_list = sorted(glob(osp.join(args.frame_dir, '*')))
    num_gpus = args.num_gpus
    chunks = [video_list[i::num_gpus] for i in range(num_gpus)]

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=process_subset, args=(args, gpu_id, chunks[gpu_id]))
        p.start()
        time.sleep(30)
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_dir', type=str, default='thumos14_frames_4fps')
    parser.add_argument('--repo_dir', type=str, default='/root/datasets/thumos14/dinov3')
    parser.add_argument('--output_dir', type=str, default='thumos14_dinov3_4fps')
    parser.add_argument('--model', type=str, default='/root/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    parser.add_argument('--num_gpus', type=int, default=4)
    args = parser.parse_args()
    
    mp.set_start_method('spawn')
    main(args)
