import jittor as jt
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# 导入你转换后的 Jittor 模型和指标类
from model.metrics import IoUMetric, nIoUMetric, PD_FA, ROCMetric2
from utils.data import SirstDataset, IRSTD1K_Dataset
from model.ilnet_jittor import ILNet_S, ILNet_M, ILNet_L 

# Jittor 自动管理设备，通过 flags 开启 CUDA
jt.flags.use_cuda = 1 if jt.has_cuda else 0

def parse_args():
    parser = ArgumentParser(description='Jittor Implementation of ILNet')
    parser.add_argument('--img_size', type=int, default=512, help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size for testing')
    parser.add_argument('--dataset', type=str, default='sirst', help='datasets: sirst or IRSTD-1k')
    parser.add_argument('--mode', type=str, default='L', help='mode: L, M, S')
    # 注意：Jittor 习惯使用 .pkl 或 .path
    parser.add_argument('--checkpoint', type=str, default='best_model.pkl', help='checkpoint path')

    args = parser.parse_args()
    return args

class Val:
    def __init__(self, args, load_path: str):
        self.args = args

        # 1. Datasets (Jittor Dataset 已经集成了 DataLoader 功能)
        if args.dataset == 'sirst':
            self.val_set = SirstDataset(args, mode='val')
        elif args.dataset == 'IRSTD-1k':
            self.val_set = IRSTD1K_Dataset(args, mode='val')
        else:
            raise NameError("Dataset not supported")
        
        # 设置 Jittor 数据集属性
        self.val_set.set_attrs(batch_size=args.batch_size, shuffle=False)

        # 2. Model 实例化
        assert args.mode in ['L', 'M', 'S']
        if args.mode == 'L':
            self.net = ILNet_L()
        elif args.mode == 'M':
            self.net = ILNet_M()
        else:
            self.net = ILNet_S()

        # 3. 加载权重
        # Jittor 直接 load 训练保存的 .pkl 文件
        self.net.load(load_path)
        self.net.eval()

        # 4. 初始化指标
        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        self.PD_FA = PD_FA(args.img_size)
        self.ROC = ROCMetric2(1, bins=10)

    def test_model(self):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()
        self.ROC.reset()

        tbar = tqdm(self.val_set)
        for i, (data, labels) in enumerate(tbar):
            # Jittor 默认不计算梯度，但在 eval 模式下更保险
            with jt.no_grad():
                output = self.net(data)
                # Jittor 不需要手动 .cpu()，直接传给 numpy 写的 metric 即可
            
            # --- 指标更新 (逻辑必须与原版一致) ---
            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)
            
            # 维度转换对齐: (N, C, H, W) -> (H, W, C)
            # Jittor 使用 transpose 替代 permute
            output2 = output.squeeze(0).transpose(1, 2, 0)
            labels2 = labels.squeeze(0)
            self.ROC.update(output2, labels2)

            # 获取当前阶段性指标
            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))

            tbar.set_description('IoU:%f, nIoU:%f, Fa:%.10f, Pd:%.10f'
                                 % (IoU, nIoU, Fa, Pd))
            
        # 计算最终 ROC 相关指标
        tpr, fpr, recall, precision = self.ROC.get()
        return IoU, nIoU, Fa, Pd, tpr, fpr

if __name__ == "__main__":
    args = parse_args()
    # 实例化并运行
    value = Val(args, load_path=args.checkpoint)
    IoU, nIoU, Fa, Pd, tpr, fpr = value.test_model()
    
    print('\n' + '='*50)
    print(f'Final Results on {args.dataset}:')
    print(f'mIoU: {IoU:.6f}')
    print(f'nIoU: {nIoU:.6f}')
    print(f'Fa:   {Fa:.10f}')
    print(f'Pd:   {Pd:.6f}')
    print(f'TPR:  {tpr}')
    print(f'FPR:  {fpr}')
    print('='*50)