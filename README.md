# ILNet (Jittor Version) 迁移实现与使用说明

本项目将原始的 PyTorch 版 ILNet 成功迁移至 **Jittor (计图)** 框架。Jittor 的元算子和统一内存管理特性，使得在相同显存下能支持更大的 Batch Size 或更高的推理效率。

## 1. Jittor 模块迁移说明

在迁移过程中，我们主要针对核心架构定义文件进行了底层适配。主要的迁移逻辑涉及以下文件：

### 核心迁移文件：

* **`model/ilnet_jittor.py`**:
* **迁移方式**：将 PyTorch 的 `nn.Module` 替换为 Jittor 的 `jt.Module`。
* **算子替换**：将所有的 `torch.nn` 算子（如 `Conv2d`, `BatchNorm2d`, `ReLU`）无缝替换为 `jt.nn` 下的对应算子。
* **池化逻辑**：针对 Jittor 的 `Pool` 算子特性，优化了特征提取部分的下采样逻辑。


* **`model/utils.py`**:
* **迁移方式**：重写了参数初始化方法。Jittor 使用不同的权重初始化 API，我们将原版的 Kaiming 初始化等策略适配到了 Jittor 算子中。



---

## 2. 新增功能文件说明

为了增强项目的易用性、监控训练过程以及方便可视化复现，我们新增了以下关键文件：

| 文件名 | 功能描述 |
| --- | --- |
| **`test_gpu.py`** | **测试Gpu环境** |
| **`heartbeat_test.py`** | **心跳测试**：初步验证模块迁移是否成功 |
| **`rebuild_logs.py`** | **指标重建工具**：针对训练中断的情况，该脚本能解析 `.txt` 格式的终端日志，提取 Loss、mIoU、Fa、Pd 等指标，并重新注入到 Tensorboard 中生成平滑曲线和 Best 阶梯图。 |
| **`visualize.py`** | **单图推理可视化**：支持加载训练好的 `.pkl` 权重，针对 SIRST 数据集（包括处理特定后缀如 `_pixels0.png`）进行推理。自动生成 **原图、GT、热力图、二值化预测图** 的四合一对比图。 |
| **`train.py` (已优化)** | **增强版训练脚本**：新增了对 Tensorboard 的原生支持，实现了断点续训功能（`--resume`），并修复了 Windows 环境下的 CUDA 同步问题。 |

---

## 3. 使用说明

### 环境准备

确保你的环境下安装了 Jittor 及其相关依赖：

```bash
pip install jittor tensorboardX

```

### 训练与可视化监控

1. **开始训练**：
```powershell
python train.py --dataset sirst --batch_size 4 --epochs 300 --save_dir ./checkpoints_final

```


2. **查看实时图表**：
在 VS Code 终端运行：
```powershell
tensorboard --logdir=./checkpoints_final/logs

```



### 结果复现可视化

如果你想查看模型在某张特定图片（如 `Misc_26`）上的检测效果：

1. 确保权重文件位于 `./checkpoints_final_finish/best_model_sirst.pkl`。
2. 运行可视化脚本：
```powershell
python visualize.py
```


3. 脚本会自动在根目录生成 `reproduced_result_Misc_26.png`。

### 日志损坏修复

如果训练过程中途崩溃且未产生 Tensorboard 事件文件：

1. 将终端打印的日志保存为 `checkpoints_final.txt`。
2. 运行 `python visualize_logs.py`。
3. 启动 Tensorboard 指向 `./checkpoints_final/logs_rebuilt` 查看完整曲线。

---

## 4. 数据集结构

本项目的 SIRST 数据集路径适配如下：

```text
datasets/SIRST/
├── images/
│   └── Misc_26.png
└── masks/
    └── Misc_26_pixels0.png  # 注意：掩码文件名包含 _pixels0 后缀

```

---

### 迁移总结

通过本次迁移，ILNet 在 Jittor 环境下实现了与 PyTorch 相同的收敛性能（mIoU 稳步上升，Fa 降至  量级），同时发挥了 Jittor 在国产算力环境下的部署优势。
