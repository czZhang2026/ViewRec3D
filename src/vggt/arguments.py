from datetime import datetime
from dataclasses import dataclass, field
from typing import List

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class DataArguments:
    """
    数据相关参数
    """
    mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )
    train_dir: str = field(
        default="assets/Dataset/train",
        metadata={"help": "训练集目录"}
    )
    val_dir: str = field(
        default="assets/Dataset/val",
        metadata={"help": "验证集目录"}
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "DataLoader 的 worker 数"}
    )
    data_ratio: float = field(
        default=1.0,
        metadata={"help": "用多少数据"}
    )
    data_angle: int = field(
        default=180,
        metadata={"help": "旋转角阈值"}
    )


@dataclass
class ModelArguments:
    """
    模型结构相关参数
    """
    ckpt_path: str = field(
        default="src/vggt/VGGT-1B/vggt.pt",
        metadata={"help": "VGGT模型权重路径"}
    )
    img_size: int = field(
        default=518,
        metadata={"help": "输入图像尺寸"}
    )
    patch_size: int = field(
        default=14,
        metadata={"help": "patch size"}
    )
    embed_dim: int = field(
        default=1024,
        metadata={"help": "embedding dimension"}
    )
    use_rays: bool = field(
        default=True,
        metadata={"help": "用rays监督"}
    )


@dataclass
class LossArguments:
    """
    多任务 loss 相关参数
    """
    change_orientation_loss_weight: float = field(
        default=1.0,
        metadata={"help": "change_orientation 分支的 loss 权重"}
    )
    pose_loss_weight: float = field(
        default=10.0,
        metadata={"help": "pose 分支的 loss 权重"}
    )
    rays_loss_weight: float = field(
        default=1.0,
        metadata={"help": "rays 分支的loss权重"}
    )


@dataclass
class VGGTTrainingArguments(TrainingArguments):
    """
    Hugging Face Trainer 参数
    """
    output_dir: str = field(
        default=datetime.now().strftime("outputs/exp_0"),
        metadata={"help": "checkpoint 和日志输出目录"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "梯度累积步数"}
    )

    logging_steps: int = field(
        default=50,
        metadata={"help": "每隔多少 step 记录一次日志"}
    )

    logging_strategy: str = field(
        default="steps",
        metadata={"help": "日志记录策略：steps 或 epoch"}
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "日志上报目标，使用 tensorboard"}
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "每张卡的训练 batch size"}
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "每张卡的验证 batch size"}
    )

    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "学习率"}
    )
    weight_decay: float = field(
        default=0.05,
        metadata={"help": "权重衰减"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "warmup 占总训练步数的比例"}
    )

    num_train_epochs: float = field(
        default=10,
        metadata={"help": "训练 epoch 数"}
    )

    save_strategy: str = field(
        default="epoch",
        metadata={"help": "checkpoint 保存策略"}
    )
    eval_strategy: str = field(
        default="epoch",
        metadata={"help": "验证策略"}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "选择最佳模型所依据的指标"}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "metric_for_best_model 是否越大越好"}
    )
    save_total_limit: int = field(
        default=2,
        metadata={"help": "最多保留多少个 checkpoint"}
    )
    ddp_find_unused_parameters: bool = field(
        default=True,
        metadata={"help": "多卡 DDP 下是否查找未使用参数；若报 unused parameters 错误可设为 True"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "训练结束后是否加载最佳模型"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "自定义模型一般应设为 False"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "是否启用 bf16 混合精度"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "是否启用 fp16 混合精度"}
    )


def parse_args():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            LossArguments,
            VGGTTrainingArguments,
        )
    )

    return parser.parse_args_into_dataclasses()