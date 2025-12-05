## 📄 Paper Info
### SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration
<p>
  <a href="https://iceclear.github.io/projects/seedvr/">
    <img
      src="https://img.shields.io/badge/SeedVR-Website-0A66C2?logo=safari&logoColor=white"
      alt="SeedVR Website"
    />
  </a>
  <a href="https://huggingface.co/collections/ByteDance-Seed/seedvr-6849deeb461c4e425f3e6f9e">
    <img 
        src="https://img.shields.io/badge/SeedVR-Models-yellow?logo=huggingface&logoColor=yellow" 
        alt="SeedVR Models"
    />
  </a>
   <a href="https://huggingface.co/spaces/ByteDance-Seed/SeedVR2-3B">
    <img 
        src="https://img.shields.io/badge/SeedVR2-Space-orange?logo=huggingface&logoColor=yellow" 
        alt="SeedVR2 Space"
    />
  </a>
  <a href="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler">
    <img
      src="https://img.shields.io/badge/SeedVR-ComfyUI-blue?logo=googleplay&logoColor=blue"
      alt="SeedVR ComfyUI"
    />
  </a>
  <a href="https://arxiv.org/abs/2501.01320">
    <img
      src="https://img.shields.io/badge/SeedVR-Paper-red?logo=arxiv&logoColor=red"
      alt="SeedVR Paper on ArXiv"
    />
  </a>
</p>


### SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training
<p>
  <a href="https://iceclear.github.io/projects/seedvr2/">
    <img
      src="https://img.shields.io/badge/SeedVR2-Website-0A66C2?logo=safari&logoColor=white"
      alt="SeedVR Website"
    />
  </a>
  <a href="https://huggingface.co/collections/ByteDance-Seed/seedvr-6849deeb461c4e425f3e6f9e">
    <img 
        src="https://img.shields.io/badge/SeedVR-Models-yellow?logo=huggingface&logoColor=yellow" 
        alt="SeedVR Models"
    />
  </a>
  <a href="https://huggingface.co/spaces/ByteDance-Seed/SeedVR2-3B">
    <img 
        src="https://img.shields.io/badge/SeedVR2-Space-orange?logo=huggingface&logoColor=yellow" 
        alt="SeedVR2 Space"
    />
  </a>
  <a href="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler">
    <img
      src="https://img.shields.io/badge/SeedVR-ComfyUI-blue?logo=googleplay&logoColor=blue"
      alt="SeedVR ComfyUI"
    />
  </a>
  <a href="http://arxiv.org/abs/2506.05301">
    <img
      src="https://img.shields.io/badge/SeedVR2-Paper-red?logo=arxiv&logoColor=red"
      alt="SeedVR2 Paper on ArXiv"
    />
  </a>
</p>


**Limitations:** These are the prototype models and the performance may not perfectly align with the paper. Our methods are sometimes not robust to heavy degradations and very large motions, and shares some failure cases with existing methods, e.g., fail to fully remove the degradation or simply generate unpleasing details. Moreover, due to the strong generation ability, Our methods tend to overly generate details on inputs with very light degradations, e.g., 720p AIGC videos, leading to oversharpened results occasionally (especially on small resolutions, e.g., 480p).


## 🔥 RTX5090 Deployment

1️⃣ Set up environment
```bash
git clone https://github.com/AaronCaoZJ/SeedVR.git
cd SeedVR
conda create -n seedvr python=3.10 -y
conda activate seedvr
pip install -r requirements.txt
# only pytorch >= 2.8.0 support RTX5090 with CUDA capability sm_120
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

Install [flash-attn](https://github.com/Dao-AILab/flash-attention)

```bash
pip install ninja
# make sure nvcc -V >> 12.8
# export PATH=/usr/local/cuda-12.8/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda-12.8
pip install flash-attn --no-build-isolation --no-cache-dir
```

Install [apex](https://github.com/NVIDIA/apex)

```bash
git clone https://github.com/NVIDIA/apex
cd apex
# make sure nvcc -V >> 12.8
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation --no-cache-dir .
# if encounter version conflict
# try add # before line 218 in `apex/setup.py`
# # check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)
```


2️⃣ Download pretrained checkpoint
```python
# Take SeedVR2-3B as an example.
# See all models: https://huggingface.co/models?other=seedvr

from huggingface_hub import snapshot_download

save_dir = "ckpts/"
repo_id = "ByteDance-Seed/SeedVR2-3B"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.pth", "*.bin", "*.py", "*.md", "*.txt"],
)
```


3️⃣ Enable color fix

Make sure file [color_fix.py](https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py) is located in `SeedVR/projects/video_diffusion_sr/color_fix.py`.


## 🔍 Inference   

```python
# Take 3B SeedVR2 model inference script as an example
torchrun --nproc-per-node=1 -m projects.inference_seedvr2_3b

# All parameters as follow
# for image upscale GPU_NUM must be 1 (--nproc-per-node=1 & set sp_size to 1)
# to modify output h & w, set res_h & res_w manually
# or change the scaler at about line 230 in `SeedVR/projects/inference_seedvr2_3b.py`
# res_h = orig_h * 2 (scaler)
# res_w = orig_w * 2
torchrun --nproc-per-node=1 -m projects/inference_seedvr2_3b.py \
    --video_path INPUT_FOLDER/run \
    --output_dir OUTPUT_FOLDER \
    --seed SEED_NUM \
    --res_h None \
    --res_w None \
    --sp_size 1
```

**Notice** Seems only jpg & jpeg, with 3 channels, can run successfully. PNG can be recognized, but errors occur with 4 channels.