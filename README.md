**For CLS token embeddings**
```
python DINOv3_extract_patch.py \
 --output_dir (output path) \
 --model (model checkpoint path) \
 --frame_dir (extracted frames in .jpg) \
 --repo_dir (path to DINOv3 repository) \
 --num_gpus (num_gpus) \
 --batch_size (batch_size) \
 --mode cls
```
**For patch embeddings**
```
python DINOv3_extract_patch.py \
 --output_dir (output path) \
 --model (model checkpoint path) \
 --frame_dir (extracted frames in .jpg) \
 --repo_dir (path to DINOv3 repository) \
 --num_gpus (num_gpus) \
 --batch_size (batch_size) \
 --mode patch
```
**Multiple model usage**
```
python DINOv3_extract_patch.py \
 --output_dir (output path 1) (output path 2) ... \
 --model (model checkpoint 1) (model checkpoint 2) ... \
 --frame_dir (extracted frames in .jpg) \
 --repo_dir (path to DINOv3 repository) \
 --num_gpus (num_gpus) \
 --batch_size (batch_size) \
 --mode cls
```
