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

def process_subset(args, gpu_id, video_subset):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    model = []
    for i in range(len(args.model)):
        REPO_DIR = args.repo_dir
        MODEL_DIR = args.model[i]
        MODEL_NAME = '_'.join(osp.basename(MODEL_DIR).split('_')[:(3 if 'convnext' in MODEL_DIR else 2)])
        model.append(torch.hub.load(REPO_DIR, MODEL_NAME, source='local', weights=MODEL_DIR).to(device).eval())
    transform = make_transform()

    save_dir = args.output_dir

    BATCH_SIZE = args.batch_size

    for video_dir in tqdm(video_subset, desc=f"GPU {gpu_id}", position=gpu_id, leave=True):
        frame_paths = sorted(glob(osp.join(video_dir, '*.jpg')))

        if not frame_paths:
            continue

        video_features = [[] for _ in range(len(args.model))]
        with torch.no_grad():
            for i in range(0, len(frame_paths), BATCH_SIZE):
                batch_paths = frame_paths[i:i + BATCH_SIZE]

                batch_tensors = []
                for img_path in batch_paths:
                    try:
                        with Image.open(img_path) as img:
                            image = img.convert("RGB")
                            batch_tensors.append(transform(image))
                    except Exception as e:
                        print(f"Warning: Skipping corrupted frame {img_path} due to {e}")

                if not batch_tensors:
                    continue

                inputs = torch.stack(batch_tensors).to(device)

                for i in range(len(args.model)):
                    if args.mode == 'cls':
                        outputs = model[i](inputs)
                    elif args.mode == 'patch':
                        outputs = model[i].forward_features(inputs)['x_norm_patchtokens']
                    video_features[i].append(outputs.cpu())

        for i in range(len(args.model)):
            result_video_features = torch.cat(video_features[i], dim=0)

            save_path = osp.join(save_dir[i], osp.basename(video_dir) + ".npy")
            np.save(save_path, result_video_features.numpy())

def main(args):
    assert len(args.output_dir) == len(args.model)
    assert args.mode in ['cls', 'patch']

    for output_dir in args.output_dir:
        os.makedirs(output_dir, exist_ok=True)
    video_list = sorted(glob(osp.join(args.frame_dir, '*')))
    num_gpus = args.num_gpus
    chunks = [video_list[i::num_gpus] for i in range(num_gpus)]

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=process_subset, args=(args, gpu_id, chunks[gpu_id]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_dir', type=str, default='/root/datasets/thumos14/frames/4fps_center')
    parser.add_argument('--repo_dir', type=str, default='/root/datasets/thumos14/features/dinov3/dinov3')
    parser.add_argument('--output_dir', nargs='+', required=True)
    parser.add_argument('--model', nargs='+', required=True)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mode', type=str, default='cls')
    args = parser.parse_args()
    
    mp.set_start_method('spawn')
    main(args)
