import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm


def frame_to_pil(frame):
    """
    将 video 中的单帧安全转换为 PIL.Image
    兼容 PIL.Image / numpy / torch.Tensor
    """
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu()
        if frame.ndim == 3:
            # 可能是 [C,H,W] 或 [H,W,C]
            if frame.shape[0] in [1, 3]:
                frame = frame.permute(1, 2, 0)
        frame = frame.numpy()

    if isinstance(frame, np.ndarray):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255)
            # 如果是 0~1，则放大到 0~255
            if frame.max() <= 1.0:
                frame = frame * 255.0
            frame = frame.astype(np.uint8)
        return Image.fromarray(frame).convert("RGB")

    raise TypeError(f"Unsupported frame type: {type(frame)}")


def resize_video_frames(video, target_size):
    """
    将整个视频帧 resize 回目标尺寸
    target_size: (width, height)
    """
    resized_frames = []
    for frame in video:
        frame_pil = frame_to_pil(frame)
        frame_pil = frame_pil.resize(target_size, Image.LANCZOS)
        resized_frames.append(frame_pil)
    return resized_frames


# Load models
model_manager = ModelManager(device="cuda")
model_manager.load_models(
    ["./checkpoints/Wan_2.1/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32,  # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "./checkpoints/Wan_2.1/diffusion_pytorch_model-00001-of-00007.safetensors",
            "./checkpoints/Wan_2.1/diffusion_pytorch_model-00002-of-00007.safetensors",
            "./checkpoints/Wan_2.1/diffusion_pytorch_model-00003-of-00007.safetensors",
            "./checkpoints/Wan_2.1/diffusion_pytorch_model-00004-of-00007.safetensors",
            "./checkpoints/Wan_2.1/diffusion_pytorch_model-00005-of-00007.safetensors",
            "./checkpoints/Wan_2.1/diffusion_pytorch_model-00006-of-00007.safetensors",
            "./checkpoints/Wan_2.1/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "./checkpoints/Wan_2.1/models_t5_umt5-xxl-enc-bf16.pth",
        "./checkpoints/Wan_2.1/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,
)
model_manager.load_lora("./checkpoints/lora/wan_lora_example.ckpt", lora_alpha=1.0)

pipe = WanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device="cuda"
)
pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)

root = Path("./assets")
input_dir = root / "input"
output_dir = root / "output"
last_frames_dir = output_dir / "images"
videos_dir = output_dir / "videos"

videos_dir.mkdir(parents=True, exist_ok=True)
last_frames_dir.mkdir(parents=True, exist_ok=True)

image_paths = []
for image_folder in input_dir.iterdir():
    if not image_folder.is_dir():
        continue
    image_path = image_folder / "generated_0.jpeg"
    if image_path.exists():
        image_paths.append(image_path)
image_paths = sorted(
    image_paths
)

if not image_paths:
    print(f"No images found in {input_dir.resolve()}")
else:
    for image_path in tqdm(image_paths):
        try:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size  # (width, height)

            video = pipe(
                prompt="improve composition, better perspective, professional photography, balanced layout, cinematic",
                negative_prompt="",
                input_image=image,
                num_inference_steps=50,
                seed=0,
                tiled=True,
            )

            # 1. 视频所有帧恢复到输入图片尺寸
            resized_video = resize_video_frames(video, original_size)

            # 2. 保存恢复尺寸后的视频
            output_video_path = videos_dir / f"{image_path.parent.stem}.mp4"
            save_video(resized_video, str(output_video_path), fps=24)

            # 3. 保存最后一帧
            last_frame = resized_video[-1]
            last_frame_path = last_frames_dir / f"{image_path.parent.stem}.jpeg"
            last_frame.save(last_frame_path)

            print(f"Saved video to {output_video_path}")
            print(f"Saved last frame to {last_frame_path}")
        except Exception as e:
            print(f"Failed on {image_path}: {e}")