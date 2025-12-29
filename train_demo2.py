import argparse
from argparse import ArgumentParser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# 导入你的模块
from div.backbones.shared import BackboneRegistry
from div.data_module_demo import SpecsDataModule
from div.sdes import SDERegistry
from div.model_demo2 import ScoreModelGAN


# 辅助函数：将参数分组，方便传给 Model 的 __init__
def get_argparse_groups(parser, args):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == "__main__":
    # 1. 基础参数解析 (用于先获取 backbone 和 sde 的名字，以便加载后续参数)
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()

    for parser_ in (base_parser, parser):
        parser_.add_argument(
            "--backbone",
            type=str,
            default="ncspp_l_crm",  # 你的U-Net
            choices=BackboneRegistry.get_all_names(),
            help="Backbone model name",
        )
        parser_.add_argument(
            "--sde",
            type=str,
            default="bridgegan",  # 你的SDE
            choices=SDERegistry.get_all_names(),
            help="SDE type",
        )
        parser_.add_argument(
            "--max_epochs",
            type=int,
            default=1000,
            help="Max training epochs.",
        )
        parser_.add_argument(
            "--gpu_id",
            type=int,
            default=0,
            help="GPU ID to use (e.g., 0).",
        )
        parser_.add_argument(
            "--seed",
            type=int,
            default=3407,
            help="Random seed.",
        )
        parser_.add_argument(
            "--exp_name",
            type=str,
            default="ocean_recon",
            help="Experiment name for logging.",
        )

    # 解析已知参数，获取 backbone 和 sde 的名字
    temp_args, _ = base_parser.parse_known_args()

    # 2. 动态加载组件类
    model_cls = ScoreModelGAN
    data_module_cls = SpecsDataModule

    # 根据名字获取类
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    sde_cls = SDERegistry.get_by_name(temp_args.sde)

    # 3. 添加各组件的特定参数到 Parser
    # Trainer 参数
    trainer_parser = parser.add_argument_group("Trainer")
    # 这里不需要再添加 max_epochs 等，因为上面已经添加了，Pl.Trainer 会自动处理剩余的 kwargs
    # 我们只添加一些特定的
    trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    trainer_parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

    # Model 参数
    model_cls.add_argparse_args(
        parser.add_argument_group("ScoreModelGAN", description="Model Hyperparams")
    )

    # SDE 参数
    sde_cls.add_argparse_args(
        parser.add_argument_group("SDE", description=f"SDE: {sde_cls.__name__}")
    )

    # Backbone 参数
    backbone_cls.add_argparse_args(
        parser.add_argument_group("BackboneScore", description=f"Backbone: {backbone_cls.__name__}")
    )

    # DataModule 参数 (train_data_dir, crop_size 等在这里被添加)
    data_module_cls.add_argparse_args(
        parser.add_argument_group("DataModule", description="Data Hyperparams")
    )

    # 4. 解析所有参数
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser, args)

    # 设置随机种子
    seed_everything(seed=args.seed, workers=True)

    # 5. 实例化 DataModule
    data_module = data_module_cls(**vars(arg_groups["DataModule"]))

    # 6. 实例化 Model
    # 注意：我们将所有分组的参数解包传递进去，model 会自己挑需要的
    model = model_cls(
        backbone=args.backbone,
        sde=args.sde,
        max_epochs=args.max_epochs,
        **{
            **vars(arg_groups["ScoreModelGAN"]),
            **vars(arg_groups["SDE"]),
            **vars(arg_groups["BackboneScore"]),
        }
    )

    # 7. 设置 Logger 和 Callbacks
    logger = TensorBoardLogger(
        save_dir="/data/zjh/bs/log/",
        name=args.exp_name,
        version=None
    )

    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        # 核心修改：监控 val_loss (MSE)，保存 Loss 最小的模型
        ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            filename="{epoch:04d}-{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=5,
            save_last=True,
        )
    ]

    # 8. 初始化 Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu_id],  # 指定 GPU
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=10,
        strategy="auto"  # 单卡通常用 auto，多卡用 ddp
    )

    print(f"Start training {args.exp_name}...")
    print(f"Data Dir: {args.train_data_dir}")

    # 9. 开始训练
    trainer.fit(model, datamodule=data_module)