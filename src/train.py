import os
import shutil
from pathlib import Path

from transformers import set_seed
import torch.distributed as dist

from src.data.builder import build_data_module
from src.trainer.trainer_builder import build_trainer
from src.vggt.arguments import parse_args
from src.vggt.builder import build_model


def main():
    # 1. 解析参数
    model_args, data_args, loss_args, training_args = parse_args()

    # 2. 随机种子
    set_seed(training_args.seed)

    if not dist.is_initialized() or dist.get_rank() == 0:
        if os.path.exists(training_args.output_dir):
            shutil.rmtree(training_args.output_dir)
    # 3. 创建输出目录
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 4. 构建数据模块
    data_module = build_data_module(
        train_dataset_dir=Path(data_args.train_dir),
        val_dataset_dir=Path(data_args.val_dir),
        seed=training_args.seed,
        target_size=model_args.img_size,
        patch_size=model_args.patch_size,
        use_rays=model_args.use_rays,
        mean=data_args.mean,
        std=data_args.std,
        data_ratio=data_args.data_ratio,
        data_angle=data_args.data_angle,
    )

    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    data_collator = data_module.data_collator

    print(f"[train.py] train samples: {len(train_dataset) if train_dataset is not None else 0}")
    print(f"[train.py] val samples: {len(val_dataset) if val_dataset is not None else 0}")

    if train_dataset is None:
        raise ValueError("train_dataset is None，无法开始训练。")

    # 5. 构建模型
    model = build_model(
        data_args=data_args,
        model_args=model_args,
        loss_args=loss_args,
    )

    # 6. 构建 Trainer
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        change_orientation_threshold=0.5,
    )

    # 7. 训练
    train_result = trainer.train()

    # 8. 保存最终模型
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # 9. 验证
    if val_dataset is not None:
        eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    print(f"[train.py] finished. outputs saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()