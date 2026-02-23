import matplotlib
matplotlib.use('Agg') # 建议保留，避免 Windows 终端由于 GUI 导致的卡死

import jittor as jt
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# 1. 导入模型
try:
    from model.ilnet_jittor import ILNet_S 
except:
    from model.ilnet import ILNet as ILNet_S

jt.flags.use_cuda = 1
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_visualize(img_name='Misc_26'):
    # --- 路径精准匹配 ---
    base_data_dir = os.path.join(ROOT_DIR, 'datasets', 'SIRST')
    
    img_path = os.path.join(base_data_dir, 'images', f'{img_name}.png')
    gt_path = os.path.join(base_data_dir, 'masks', f'{img_name}_pixels0.png')
    
    # 权重路径：指向训练完成后的 best 权重
    model_path = os.path.join(ROOT_DIR, 'checkpoints_final', 'best_model_sirst.pkl')

    # 路径存在性检查
    print(f"--- 正在检查路径 ---")
    if not os.path.exists(img_path):
        print(f"❌ 找不到原图: {img_path}")
        return
    if not os.path.exists(gt_path):
        print(f"❌ 找不到掩码: {gt_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ 找不到权重: {model_path}")
        return
    print(f"✅ 路径检查通过，开始推理...")

    # 2. 加载模型与权重
    model = ILNet_S(out_ch=1)
    model.load(model_path)
    model.eval()

    # 3. 图像预处理
    # 保持与训练一致的尺寸 (512, 512)
    raw_img_pil = Image.open(img_path).convert('RGB').resize((512, 512))
    img_array = np.array(raw_img_pil).transpose(2, 0, 1) / 255.0
    input_tensor = jt.array(img_array).float().unsqueeze(0)
    
    # 4. 模型推理
    with jt.no_grad():
        output = model(input_tensor)
        # 结果需要过 Sigmoid 映射到 [0, 1] 置信度
        pred = jt.sigmoid(output).numpy()[0, 0]

    # 5. 读取 Ground Truth
    gt_img = Image.open(gt_path).convert('L').resize((512, 512))
    
    # 6. 生成二值图 (阈值设为 0.5)
    pred_binary = (pred > 0.5).astype(np.uint8)

    print(f"📊 推理完成！预测得分最大值: {pred.max():.4f}")

    # 7. 绘图与保存
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1); plt.imshow(raw_img_pil); plt.title('Original Image'); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(gt_img, cmap='gray'); plt.title('Ground Truth'); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(pred, cmap='jet'); plt.title('Prediction Heatmap'); plt.axis('off')
    plt.subplot(1, 4, 4); plt.imshow(pred_binary, cmap='gray'); plt.title('Binary Result'); plt.axis('off')

    plt.tight_layout()
    
    # 保存结果到当前脚本目录下
    save_filename = f'reproduced_result_{img_name}.png'
    plt.savefig(os.path.join(ROOT_DIR, save_filename))
    print(f"🎉 可视化复现成功！结果已保存为: {save_filename}")

if __name__ == "__main__":
    # 执行
    run_visualize('Misc_1')