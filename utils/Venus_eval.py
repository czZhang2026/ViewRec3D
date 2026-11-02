import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_box_from_response(response: str) -> tuple[int, int, int, int]:
    """
    从模型输出中解析 box 坐标。
    示例:
    <ref>the most visually balanced and aesthetically pleasing composition area</ref><box>(1,1),(1000,960)</box>
    """
    pattern = r"<box>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*</box>"
    match = re.search(pattern, response)

    if match is None:
        raise ValueError(f"未能从 response 中解析出 box:\n{response}")

    x1, y1, x2, y2 = map(int, match.groups())
    return x1, y1, x2, y2


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """
    将 box 限制在图像范围内，并保证 x1 < x2, y1 < y2
    """
    x1, y1, x2, y2 = box

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"裁剪框非法，处理后得到: {(x1, y1, x2, y2)}，图像尺寸: {(width, height)}")

    return x1, y1, x2, y2


def crop_image(image_path: str, box: tuple[int, int, int, int], save_path: str) -> None:
    """
    按照 box 裁剪图片并保存
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    box = clamp_box(box, width, height)
    cropped = image.crop(box)
    cropped.save(save_path)


def main():
    model_path = "./checkpoints/Venus-Q-Stage2"
    root_path = Path("./assets/input")

    output_dir = Path("./assets/output")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        bf16=True,
    ).eval()

    for image_folder in tqdm(root_path.iterdir()):
        if not image_folder.is_dir():
            continue
        image_path = image_folder / "generated_0.jpeg"
        query = tokenizer.from_list_format([
            {"image": str(image_path)},
            {
                "text": "Please provide the bounding box coordinate of the most visually balanced and aesthetically pleasing composition area."
            },
        ])

        response, history = model.chat(tokenizer, query=query, history=None)

        # 解析 box
        box = parse_box_from_response(response)


        crop_path = output_dir / f"{image_folder.name}.jpeg"
        crop_image(str(image_path), box, str(crop_path))


if __name__ == "__main__":
    main()