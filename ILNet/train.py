import jittor as jt
jt.flags.use_cuda = 1
# 针对 RTX 40 系列显卡的兼容性优化
#jt.flags.nvcc_flags = " --use_fast_math"
from jittor import nn, optim
from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as osp
import numpy as np

# 导入转换后的模块
from model.ilnet_jittor import ILNet_S 
from model.loss_jittor import SoftIoULoss, criterion
from utils.data import SirstDataset, IRSTD1K_Dataset
from model.metrics import IoUMetric, nIoUMetric, PD_FA 

def parse_args():
    parser = ArgumentParser(description='ILNet Training with Jittor')
    parser.add_argument('--dataset', type=str, default='sirst', choices=['sirst', 'irstd1k'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300, help='Total epochs to run')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    # --- 新增通用控制参数 ---
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='manual epoch number (useful on restarts)')
    # ----------------------
    
    return parser.parse_args()

def train():
    args = parse_args()
    jt.flags.use_cuda = 1 
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 1. 数据集加载
    if args.dataset == 'sirst':
        train_dataset = SirstDataset(args, mode='train')
        val_dataset = SirstDataset(args, mode='val')
    else:
        train_dataset = IRSTD1K_Dataset(args, mode='train')
        val_dataset = IRSTD1K_Dataset(args, mode='val')

    train_dataset.set_attrs(batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataset.set_attrs(batch_size=1, shuffle=False)

    # 2. 模型、损失函数、优化器
    model = ILNet_S()
    iou_loss_func = SoftIoULoss()
    
    # 3. 权重恢复逻辑
    best_iou = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"===> Loading checkpoint from {args.resume}...")
            model.load(args.resume)
            # 可以在这里根据经验手动设一个 best_iou 门槛，或者让它从0重新竞争
        else:
            print(f"===> Warning: no checkpoint found at {args.resume}")

    # 注意：Adam 优化器需要在加载模型后再实例化，以确保参数同步
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # 4. 训练循环
    for epoch in range(args.start_epoch, args.epochs): 
        model.train()
        train_loss = []
        tbar = tqdm(train_dataset)
        
        # 比如：每到总进度的 50% 和 75% 时衰减
        if epoch == int(args.epochs * 0.5) or epoch == int(args.epochs * 0.75):
            optimizer.lr *= 0.1
            print(f"--- Learning rate decayed to {optimizer.lr} ---")
        for i, (images, masks) in enumerate(tbar):
            output = model(images)
            loss_bce = criterion(output, masks)
            loss_iou = iou_loss_func(output, masks)
            total_loss = loss_bce + loss_iou
            
            optimizer.step(total_loss)
            
            train_loss.append(total_loss.item())
            tbar.set_description(f"Epoch [{epoch}/{args.epochs}] Loss: {np.mean(train_loss):.4f} LR: {optimizer.lr:.6f}")

        # 5. 验证与保存
        if (epoch + 1) % 5 == 0 or epoch == 0:
            mIoU, nIoU, fa, pd = validate(model, val_dataset, args.img_size)
            print(f"Eval @ Epoch {epoch}: mIoU: {mIoU:.4f}, nIoU: {nIoU:.4f}, Fa: {fa:.10f}, Pd: {pd:.4f}")
            
            if mIoU > best_iou:
                best_iou = mIoU
                save_path = osp.join(args.save_dir, f"best_model_{args.dataset}.pkl")
                model.save(save_path)
                print(f"--- Best model saved (mIoU: {best_iou:.4f}) ---")

def validate(model, val_dataset, img_size):
    model.eval()
    metric_iou = IoUMetric()
    metric_niou = nIoUMetric()
    metric_pdfa = PD_FA(img_size)
    for images, masks in val_dataset:
        with jt.no_grad():
            output = model(images)
            metric_iou.update(output, masks)
            metric_niou.update(output, masks)
            metric_pdfa.update(output, masks)
    _, mIoU = metric_iou.get()
    _, nIoU = metric_niou.get()
    fa, pd = metric_pdfa.get(len(val_dataset))
    return mIoU, nIoU, fa, pd

if __name__ == "__main__":
    train()