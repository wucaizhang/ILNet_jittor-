import jittor as jt
jt.flags.use_cuda = 1

print(f"Jittor 版本: {jt.__version__}")
print(f"当前是否启用 CUDA: {jt.flags.use_cuda}")

# 查看 Jittor 识别到的 GPU 架构
from jittor import compiler
print(f"Jittor 探测到的设备架构: {jt.compile_extern.cuda_archs}")

try:
    # 强制进行一次显存分配同步
    a = jt.array([1, 2, 3]).cuda()
    # 执行一个简单的加法触发计算核心
    b = a + a
    print(f"计算结果: {b.numpy()}")
    print("✅ 完美！RTX 4060 已经可以正常进行算子计算。")
except Exception as e:
    print(f"❌ 运行报错详情: {e}")