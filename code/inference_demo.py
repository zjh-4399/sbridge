import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from div.model_demo2 import ScoreModelGAN

def load_txt_data(file_path):
    return np.loadtxt(file_path, dtype=np.float32)

class ModelAdapter(torch.nn.Module):
    def __init__(self, target_model):
        super().__init__()
        self.target_model = target_model
    
    def forward(self, x, cond, t):
        # 适配器：调整参数顺序 (x, cond, t) -> (x, t, cond)
        return self.target_model(x, t, cond)

def predict_sliding_window(model, input_tensor, crop_size=128, overlap=0.5):
    """
    带重叠(Overlap)的滑动窗口预测 - 完整修复版
    """
    b, c, h, w = input_tensor.shape
    
    # 1. 🔥 关键：获取当前设备 (cuda 或 cpu)
    device = input_tensor.device 
    
    # 2. 高度处理 (补齐到 128)
    pad_h = 0
    if h < crop_size:
        pad_h = crop_size - h
        input_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, pad_h), value=0)
    
    # 3. 准备输出容器
    output_tensor = torch.zeros_like(input_tensor)
    count_map = torch.zeros_like(input_tensor)
    
    # 计算步长
    stride = int(crop_size * (1 - overlap)) 
    
    adapter = ModelAdapter(model)
    print(f"Starting sliding window (Width: {w}, Stride: {stride}, Device: {device})...")
    
    current_w = 0
    while current_w < w:
        start_w = current_w
        end_w = current_w + crop_size
        
        # 边界处理
        if end_w > w:
            end_w = w
            start_w = w - crop_size
        
        # 截取
        slice_input = input_tensor[..., start_w:end_w]
        
        # 预测
        with torch.no_grad():
            # 这里可以临时增加步数 model.sde.N = 50 以获得更好效果
            model.sde.N = 4
            slice_output = model.sde.reverse_diffusion(
                x1=slice_input,
                cond=slice_input,
                dnn=adapter
            )
        
        slice_output = slice_output.to(device)
        
        # 累加
        output_tensor[..., start_w:end_w] += slice_output
        count_map[..., start_w:end_w] += 1.0
        
        print(f"  Processed patch: {start_w} - {end_w}")
        
        # 移动窗口
        if end_w == w:
            break
        current_w += stride

    # 4. 取平均
    output_tensor = output_tensor / count_map
    
    # 5. 裁剪掉补的高度
    if pad_h > 0:
        output_tensor = output_tensor[..., :h, :]
        
    return output_tensor

def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    print(f"Loading model from: {args.ckpt_path}")
    model = ScoreModelGAN.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.to(device)

    # 2. 读取数据
    print(f"Reading data...")
    low_r = load_txt_data(os.path.join(args.input_dir, f"{args.sample_id}_low_real.txt"))
    low_i = load_txt_data(os.path.join(args.input_dir, f"{args.sample_id}_low_imag.txt"))
    
    high_r_path = os.path.join(args.input_dir, f"{args.sample_id}_high_real.txt")
    if os.path.exists(high_r_path):
        high_r = load_txt_data(high_r_path)
        high_i = load_txt_data(os.path.join(args.input_dir, f"{args.sample_id}_high_imag.txt"))
        ref_data = np.stack([high_r, high_i])
        has_gt = True
    else:
        print("Warning: High-res Ground Truth not found. Using Low-res for normalization stats.")
        ref_data = np.stack([low_r, low_i])
        has_gt = False

    # 3. 预处理
    low_res_np = np.stack([low_r, low_i], axis=0)
    
    val_min = np.min(ref_data)
    val_max = np.max(ref_data)
    denominator = val_max - val_min if (val_max - val_min) > 1e-8 else 1.0
    
    low_res_norm = 2.0 * (low_res_np - val_min) / denominator - 1.0
    low_res_tensor = torch.from_numpy(low_res_norm).unsqueeze(0).to(device)

    # 4. 执行滑动窗口生成
    # 注意：这里调用了修复后的函数
    generated_tensor = predict_sliding_window(model, low_res_tensor, crop_size=128, overlap=0.5)

    # 5. 后处理
    generated_np = generated_tensor.cpu().numpy().squeeze(0)
    generated_denorm = (generated_np + 1.0) * denominator / 2.0 + val_min

    # 6. 保存与可视化
    os.makedirs(args.output_dir, exist_ok=True)
    
    save_r = generated_denorm[0]
    save_i = generated_denorm[1]
    np.savetxt(os.path.join(args.output_dir, f"{args.sample_id}_pred_real.txt"), save_r)
    np.savetxt(os.path.join(args.output_dir, f"{args.sample_id}_pred_imag.txt"), save_i)
    print(f"Saved .txt results to {args.output_dir}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input (Low Res)")
    plt.imshow(low_r, cmap='jet', aspect='auto')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Generated (Sliding Window)")
    plt.imshow(save_r, cmap='jet', aspect='auto')
    plt.colorbar()

    if has_gt:
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(high_r, cmap='jet', aspect='auto')
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.sample_id}_comparison_sliding.png"))
    print(f"Saved visualization to {os.path.join(args.output_dir, f'{args.sample_id}_comparison_sliding.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--sample_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    main(args)