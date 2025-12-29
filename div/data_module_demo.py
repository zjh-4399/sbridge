import os
import glob
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class PressureTxtDataset(Dataset):
    def __init__(self, data_dir, mode='train', crop_size=2048, target_height=128):
        self.data_dir = data_dir
        self.mode = mode
        self.crop_size = crop_size  # 这里应为 2048
        self.target_height = target_height

        # 搜索目录下所有的低分实部文件
        search_pattern = os.path.join(data_dir, "*_low_real.txt")
        self.files_low_r = sorted(glob.glob(search_pattern))
        
        if len(self.files_low_r) == 0:
            print(f"Warning: No files found in {data_dir} with pattern *_low_real.txt")

        # 提取 ID
        self.ids = [os.path.basename(f).replace("_low_real.txt", "") for f in self.files_low_r]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cur_id = self.ids[idx]
        
        # 1. 构造路径
        f_low_r = os.path.join(self.data_dir, f"{cur_id}_low_real.txt")
        f_low_i = os.path.join(self.data_dir, f"{cur_id}_low_imag.txt")
        f_high_r = os.path.join(self.data_dir, f"{cur_id}_high_real.txt")
        f_high_i = os.path.join(self.data_dir, f"{cur_id}_high_imag.txt")
        
        # 2. 加载数据
        try:
            low_r = np.loadtxt(f_low_r, dtype=np.float32)
            low_i = np.loadtxt(f_low_i, dtype=np.float32)
            high_r = np.loadtxt(f_high_r, dtype=np.float32)
            high_i = np.loadtxt(f_high_i, dtype=np.float32)
        except Exception as e:
            print(f"Error reading {cur_id}: {e}")
            return torch.zeros(2, self.target_height, self.crop_size), torch.zeros(2, self.target_height, self.crop_size), cur_id

        # 3. 堆叠 (2, H, W)
        low_res = np.stack([low_r, low_i], axis=0)
        high_res = np.stack([high_r, high_i], axis=0)
        
        # 4. 归一化 (Min-Max -> [-1, 1])
        val_min = np.min(high_res)
        val_max = np.max(high_res)
        denominator = val_max - val_min if (val_max - val_min) > 1e-8 else 1.0
        
        low_res = 2.0 * (low_res - val_min) / denominator - 1.0
        high_res = 2.0 * (high_res - val_min) / denominator - 1.0
        
        low_res = torch.from_numpy(low_res)
        high_res = torch.from_numpy(high_res)
        
        # 5. Padding (高度补零) - 确保高度是 128
        H = low_res.shape[1]
        if H < self.target_height:
            pad_bottom = self.target_height - H
            low_res = torch.nn.functional.pad(low_res, (0, 0, 0, pad_bottom), value=-1)
            high_res = torch.nn.functional.pad(high_res, (0, 0, 0, pad_bottom), value=-1)
        
        # 6. 宽度处理 (固定 2048，超长截断，不足补零)
        # 所有的模式（train/val）都统一处理
        target_w = self.crop_size 
        W = low_res.shape[-1]
        
        if W >= target_w:
            # 超过长度：直接截取前 target_w 个点
            low_res = low_res[:, :, :target_w]
            high_res = high_res[:, :, :target_w]
        else:
            # 不足长度：右侧补零
            pad_right = target_w - W
            low_res = torch.nn.functional.pad(low_res, (0, pad_right, 0, 0), value=-1)
            high_res = torch.nn.functional.pad(high_res, (0, pad_right, 0, 0), value=-1)
        
        return high_res, low_res, cur_id

class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_data_dir", type=str, required=True)
        parser.add_argument("--val_data_dir", type=str, required=True)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--crop_size", type=int, default=2048) # 默认改为 2048
        
        # 哑参数
        dummy_args = [
            "dataset_name", "raw_wavfile_path", "sampling_rate", "n_fft", 
            "num_mels", "hop_size", "win_size", "fmin", "fmax", "num_frames",
            "phase_init", "spec_factor", "spec_abs_exponent", "transform_type", "format"
        ]
        for arg in dummy_args:
            parser.add_argument(f"--{arg}", default=0, help="Unused dummy arg")
        
        parser.add_argument("--normalize", type=bool, default=True, help="Unused")
        parser.add_argument("--drop_last_freq", type=bool, default=True, help="Unused")
        
        return parser

    def __init__(self, train_data_dir, val_data_dir, batch_size=4, num_workers=4, crop_size=2048, **kwargs):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = PressureTxtDataset(self.train_data_dir, mode='train', crop_size=self.crop_size)
            self.valid_set = PressureTxtDataset(self.val_data_dir, mode='val', crop_size=self.crop_size)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)