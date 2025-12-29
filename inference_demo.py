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
        return self.target_model(x, t, cond)

def predict_fixed_length(model, input_tensor, target_width=2048, target_height=128):
    """
    固定长度预测：只处理前 2048 个点
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device 
    
    # --- 1. 高度处理 (固定补齐到 128) ---
    pad_h = 0
    if h < target_height:
        pad_h = target_height - h
        input_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, pad_h), value=-1)
    
    # --- 2. 宽度处理 (固定 2048) ---
    original_w = w
    pad_w = 0
    
    if w >= target_width:
        # 超过 2048 -> 截断
        input_tensor = input_tensor[..., :target_width]
        print(f"Input width {w} > {target_width}, truncated to {target_width}.")
    else:
        # 不足 2048 -> 补零
        pad_w = target_width - w
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, 0), value=-1)
        print(f"Input width {w} < {target_width}, padded to {target_width}.")

    # --- 3. 推理 ---
    adapter = ModelAdapter(model)
    
    with torch.no_grad():
        # 这里建议使用较大的 N (如 100) 以保证质量，除非你确定模型支持极速生成
        model.sde.N = 100  
        output_tensor = model.sde.reverse_diffusion(
            x1=input_tensor,
            cond=input_tensor,
            dnn=adapter
        )
    
    # --- 4. 后处理 (裁剪掉 Padding 部分) ---
    # 高度还原
    if pad_h > 0:
        output_tensor = output_tensor[..., :h, :]
    
    # 宽度处理：用户要求"推理也只处理前2048个点"，意味着输出最大就是 2048
    # 如果原始输入小于 2048 (例如 1000)，我们把补零的那部分去掉，还原回 1000
    if pad_w > 0:
        real_w = target_width - pad_w
        output_tensor = output_tensor[..., :real_w]
    
    return output_tensor

def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    print(f"Loading model from: {args.ckpt_path}")
    model = ScoreModelGAN.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.sde.N = 50
    model.to(device)

    # 2. 读取数据
    print(f"Reading data...")
    try:
        low_r = load_txt_data(os.path.join(args.input_dir, f"{args.sample_id}_low_real.txt"))
        low_i = load_txt_data(os.path.join(args.input_dir, f"{args.sample_id}_low_imag.txt"))
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

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

    # 4. 执行固定长度预测 (2048)
    # 强制 target_width=2048
    generated_tensor = predict_fixed_length(
        model, 
        low_res_tensor, 
        target_width=2048, 
        target_height=128
    )

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

    # 绘图部分
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input (Low Res)")
    plt.imshow(low_r[:, 0:2047], cmap='jet', aspect='auto')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Generated (Fixed 2048)")
    plt.imshow(save_r, cmap='jet', aspect='auto')
    plt.colorbar()

    if has_gt:
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(high_r[:, 0:2047], cmap='jet', aspect='auto')
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.sample_id}_comparison_fixed.png"))
    print(f"Saved visualization to {os.path.join(args.output_dir, f'{args.sample_id}_comparison_fixed.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--sample_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    main(args)