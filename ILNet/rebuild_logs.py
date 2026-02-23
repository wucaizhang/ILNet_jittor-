import re
from tensorboardX import SummaryWriter
import os

log_dir = "./checkpoints_final/logs_full_best"
writer = SummaryWriter(log_dir)
file_path = "checkpoints_final.txt" 

best_metrics = {
    'mIoU': 0.0, 
    'nIoU': 0.0, 
    'Pd': 0.0, 
    'Fa': 1.0
}

if not os.path.exists(file_path):
    print(f"❌ 错误：找不到文件 {file_path}")
    exit()

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("正在解析全指标（含 Best 阶梯线）数据...")

for line in lines:
    # 匹配训练 Loss
    train_match = re.search(r'Epoch \[\s*(\d+)/300\].*?Loss:\s*([\d\.]+)', line)
    if train_match:
        epoch, loss = int(train_match.group(1)), float(train_match.group(2))
        writer.add_scalar('Train/Loss', loss, epoch)

    # 匹配评估指标
    eval_match = re.search(r'Eval @ Epoch (\d+):\s*mIoU:\s*([\d\.]+),\s*nIoU:\s*([\d\.]+),\s*Fa:\s*([\d\.\-e]+),\s*Pd:\s*([\d\.]+)', line)
    
    if eval_match:
        epoch = int(eval_match.group(1))
        cur_vals = {
            'mIoU': float(eval_match.group(2)),
            'nIoU': float(eval_match.group(3)),
            'Fa': float(eval_match.group(4)),
            'Pd': float(eval_match.group(5))
        }

        # 1. 写入所有原始波动的曲线
        for key, val in cur_vals.items():
            writer.add_scalar(f'Eval/{key}', val, epoch)

        # 2. 计算并写入 Best 值 (mIoU, nIoU, Pd 找最大)
        for key in ['mIoU', 'nIoU', 'Pd']:
            best_metrics[key] = max(best_metrics[key], cur_vals[key])
            writer.add_scalar(f'Best/{key}', best_metrics[key], epoch)
        
        # 3. 计算并写入 Best Fa (过滤 0，找最小)
        cur_fa = cur_vals['Fa']
        if cur_fa > 0:
            if cur_fa < best_metrics['Fa']:
                best_metrics['Fa'] = cur_fa
            # 只有获取到有效 Fa 后才写入 Best 分类
            writer.add_scalar('Best/Fa', best_metrics['Fa'], epoch)

writer.close()
print(f"✅ 处理完成！Best/Fa 现在已过滤第 0 轮的干扰。")
print("👉 查看指令: tensorboard --logdir=./checkpoints_final/logs_full_best")