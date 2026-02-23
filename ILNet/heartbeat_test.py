import jittor as jt
from jittor import nn
import numpy as np

# 确保开启 CUDA
jt.flags.use_cuda = 1

def test():
    try:
        # 1. 导入转换后的模型 (假设文件名为 ilnet_jittor.py)
        from model.ilnet_jittor import ILNet_S
        from model.loss_jittor import SoftIoULoss, criterion
        
        print("--- Step 1: Model Initialization ---")
        model = ILNet_S()
        
        # 2. 构造随机输入数据 [Batch, Channel, H, W]
        # 红外图像通常是 3 通道 (RGB 模式读取)
        dummy_input = jt.randn((2, 3, 512, 512))
        dummy_target = jt.randint(0, 2, (2, 1, 512, 512)).float32()
        
        print(f"Input shape: {dummy_input.shape}")
        
        # 3. 前向传播测试
        print("\n--- Step 2: Forward Pass ---")
        output = model(dummy_input)
        if isinstance(output, list):
            print(f"Output is a list of {len(output)} tensors. Side 0 shape: {output[0].shape}")
        else:
            print(f"Output shape: {output.shape}")
            
        # 4. 损失函数测试
        print("\n--- Step 3: Loss Calculation ---")
        iou_loss_func = SoftIoULoss()
        loss_bce = criterion(output, dummy_target)
        loss_iou = iou_loss_func(output[0] if isinstance(output, list) else output, dummy_target)
        total_loss = loss_bce + loss_iou
        print(f"BCE Loss: {loss_bce.item():.4f}, IoU Loss: {loss_iou.item():.4f}")
        
        # 5. 反向传播与优化器测试
        print("\n--- Step 4: Backward Pass & Optimizer ---")
        optimizer = nn.Adam(model.parameters(), lr=0.001)
        optimizer.step(total_loss)
        print("Optimizer step successful!")
        
        print("\n✅ Heartbeat Test Passed!")
        
    except Exception as e:
        print(f"\n❌ Heartbeat Test Failed!")
        print(f"Error Message: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()