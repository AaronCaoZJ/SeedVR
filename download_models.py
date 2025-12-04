from huggingface_hub import snapshot_download

# snapshot_download(
#   repo_id="openvla/openvla-7b-finetuned-libero-spatial",
#   repo_type="model",
# #   local_dir="/desired/models/storage/path"
# )

# snapshot_download(
#     repo_id="a686d380/h-corpus-2023",
#     repo_type="dataset",  # 注意这里改为 dataset
#     local_dir="/home/zhijun/datasets/h-corpus-2023"
# )


# Take SeedVR2-3B as an example.
# See all models: https://huggingface.co/models?other=seedvr

from huggingface_hub import snapshot_download

save_dir = "/home/zhijun/Code/SeedVR/ckpts/"
repo_id = "ByteDance-Seed/SeedVR2-7B"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.pth", "*.bin", "*.py", "*.md", "*.txt"],
)
