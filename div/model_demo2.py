import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import warnings

# 保留必要的 Registry 和 SDE
from div.sdes import SDERegistry
from div.backbones import BackboneRegistry
from torch.optim.lr_scheduler import CosineAnnealingLR

class ScoreModelGAN(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--opt_type", type=str, choices=['Adam', 'AdamW'], default='AdamW',
                            help='The optimizer type.')
        parser.add_argument("--lr", type=float, default=2e-4, 
                            help="The learning rate")
        parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam")
        parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Adam")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
        parser.add_argument("--t_eps", type=float, default=0.03, help="Minimum time step")
        parser.add_argument("--loss_type", type=str, default="mse", help="Loss type (mse/mae)")
        return parser

    def __init__(self,
        backbone: str = "ncsnpp", # 你的 Backbone 名字
        sde: str = "bridgegan",
        opt_type: str = 'AdamW',
        lr: float = 2e-4, 
        beta1: float = 0.9,
        beta2: float = 0.999,
        ema_decay: float = 0.999,
        t_eps: float = 3e-2, 
        nolog: bool = False,
        max_epochs = 3100,
        **kwargs # 接收 data_module_cls 等旧参数但不使用，防止报错
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1. 初始化 SDE (Schrödinger Bridge 或 OUVESDE)
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        self.sde_name = sde.lower()

        # 2. 初始化 Backbone (U-Net)
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        # input_channels=4: 因为输入是 xt(2通道) + y(2通道) = 4通道
        # 你的数据是 (实部, 虚部)，所以是 2 个通道
        # 我们在这里强制更新 kwargs 里的 input_channels
        kwargs['input_channels'] = 4
        self.dnn = dnn_cls(**kwargs)

        # 3. 优化器参数
        self.opt_type = opt_type
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_epochs = max_epochs

        # 4. EMA (指数移动平均)，有助于模型稳定
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        
        self.t_eps = t_eps
        self.nolog = nolog

    def on_train_start(self):
        if self.ema is not None:
            # 将 EMA 的影子参数移动到模型所在的 device (GPU)
            self.ema.to(self.device)

    def configure_optimizers(self):
        if self.opt_type == "Adam":
            optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif self.opt_type == "AdamW":
            optimizer = torch.optim.AdamW(self.dnn.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        else:
            raise ValueError("Unknown optimizer")

        # 使用余弦退火学习率调度
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_step(self, *args, **kwargs):
        # 每次更新参数后，同步更新 EMA 参数
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    def on_load_checkpoint(self, checkpoint):
        # 加载 EMA 权重
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        # 保存 EMA 权重
        checkpoint['ema'] = self.ema.state_dict()

    def forward(self, x, t, y, **kwargs):
        """
        前向传播
        x: 当前状态 xt (B, 2, H, W)
        t: 时间步 (B,)
        y: 条件/低分辨率输入 (B, 2, H, W)
        """
        # U-Net 预测 Score 或 Drift
        score = self.dnn(x, cond=y, time_cond=t)
        return score

    def _step(self, batch, batch_idx):
        """
        核心训练步骤
        """
        # 1. 解包数据 (对应 data_module.py 的返回)
        # x: High-Res (Target) (B, 2, H, W)
        # y: Low-Res (Condition) (B, 2, H, W)
        # id: 样本ID (B,)
        x, y, _ = batch

        # 2. 采样时间步 t
        # 在 [t_eps, 1-t_eps] (BridgeGAN) 或 [t_eps, T] (OUVE) 之间采样
        if self.sde_name == 'bridgegan':
            t = torch.rand(x.shape[0], device=x.device)
            t = torch.clamp(t, self.sde.offset, 1.0 - self.sde.offset)
        else:
            t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps

        # 3. 前向扩散 (Forward Diffusion)
        # 计算中间状态 xt 和 训练目标 target
        # x0=HighRes, x1=LowRes
        if self.sde_name == 'ouve':
            xt, target, std = self.sde.forward_diffusion(x0=x, x1=y, t=t)
        else: # bridgegan
            xt, target = self.sde.forward_diffusion(x0=x, x1=y, t=t)

        # 4. 模型预测
        pred = self(xt, t, y)

        # 5. 计算 Loss (MSE)
        if self.sde_name == 'ouve':
             # OUVE loss 逻辑
             err = pred * std + target 
             loss = torch.mean(torch.square(err))
        else:
            # BridgeGAN: pred 试图拟合 target (Drift / Score)
            err = pred - target
            loss = torch.mean(torch.square(err))

        loss_dict = {"mse": loss.item()}
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._step(batch, batch_idx)
        # 记录日志
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证集只计算 Loss，不更新参数
        loss, loss_dict = self._step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # EMA 相关逻辑：在 Eval 模式下自动切换为 EMA 权重
    def train(self, mode=True, no_ema=False):
        res = super().train(mode)
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # 进入 eval 模式：保存当前参数，加载 EMA 参数
                self.ema.store(self.dnn.parameters())
                self.ema.copy_to(self.dnn.parameters())
            else:
                # 回到 train 模式：恢复原来的参数
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def load_score_model(self, checkpoint_path):
        print(f"Loading score model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            k_new = k.replace("module.", "").replace("dnn.", "")
            new_state_dict[k_new] = v
            

        self.dnn.load_state_dict(new_state_dict, strict=False)