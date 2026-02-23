import jittor as jt
from jittor import nn

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def execute(self, pred, target):
        # Jittor 的 sigmoid 可以直接调用
        pred = jt.sigmoid(pred)
        smooth = 1
        intersection = pred * target

        # 修复点：关键字必须是 dims
        intersection_sum = jt.sum(intersection, dims=(1, 2, 3))
        pred_sum = jt.sum(pred, dims=(1, 2, 3))
        target_sum = jt.sum(target, dims=(1, 2, 3))
        
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - jt.mean(loss)
        return loss

def criterion(inputs, target):
    # 定义 BCE 损失函数
    bce_loss = nn.BCEWithLogitsLoss()
    
    if isinstance(inputs, list):
        # 如果模型有多个输出（side outputs）
        losses = [bce_loss(inputs[i], target) for i in range(len(inputs))]
        total_loss = sum(losses)
    else:
        # 单输出
        total_loss = bce_loss(inputs, target)

    return total_loss