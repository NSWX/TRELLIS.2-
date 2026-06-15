# TRELLIS.2 边缘计算全离线部署规范 (Ubuntu 22.04)

本指南针对提供 TRELLIS.2 [TRELLIS.2]:https://github.com/user-attachments/assets/63b43a7e-acc7-4c81-a900-6da450527d8f 的 100% 物理级离线环境构建与算子编译流程。



## ⚙️ 系统与依赖兼容性

* **OS**: Ubuntu 22.04 LTS
* **Python**: 3.10
* **CUDA Toolkit**: 12.4 (强隔离模式)
* **PyTorch**: 2.6.0+cu124
* **底层算子**: flash-attn==2.7.3, flex_gemm, nvdiffrast, nvdiffrec, cumesh, o-voxel

---

## 🛠️ Step 1: 虚拟环境沙盒构建与物理隔离

严禁使用全局宿主机环境，必须通过 `micromamba` 构建纯净沙盒，并注入隔离变量以斩断 `~/.local/` 目录下的旧版 NumPy/Pandas 污染。

```bash
# 1. 创建并激活沙盒
micromamba create -n trellis2 python=3.10 -y
micromamba activate trellis2

# 2. 物理级切断全局变量寻址 (防止 ABI Incompatibility)
export PYTHONNOUSERSITE=1
echo 'export PYTHONNOUSERSITE=1' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 3. 注入虚拟环境专属 CUDA 12.4 编译工具链
micromamba install -c nvidia cuda-toolkit=12.4 -y
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
```

## 🧠Step 2: 核心推理引擎与自定义算子编译

必须强制对齐 PyTorch 版本与 CUDA Toolkit 版本，防止 C++ 扩展编译时抛出版本撕裂错误。

Bash

```
# 1. 强制安装 CUDA 12.4 原生血统的 PyTorch
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 2. 清理残骸并补齐编译期刚需依赖
rm -rf /tmp/extensions/
pip install psutil

# 3. 越过隔离沙盒手动编译 flash-attn
pip install flash-attn==2.7.3 --no-build-isolation

# 4. 执行官方构建脚本 (挂载国内镜像加速剩余 Python 依赖)
. ./setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

## 📦 Step 3: 前处理与特征提取模型强制本地化

使用底层 `curl` 结合 SOCKS5 代理穿透（若需）及 HF Token 进行物理级下载，规避 `httpx` SSL 握手超时。

Bash

```
# 设置网络环境 (依据实际代理端口修改)
export HF_ENDPOINT=[https://hf-mirror.com]
export HF_TOKEN="hf_您的真实Read权限Token"

# 1. 下载 DinoV3 (受限模型，必须带 Token 头)
hf download camenduru/dinov3-vitl16-pretrain-lvd1689m --local-dir dinov3-vitl16-pretrain-lvd1689m

# 2. 下载 RMBG-2.0 (受限模型，必须带 Token 头)
hf download briaai/RMBG-2.0 --local-dir briaai
```

## 🧩 Step 4: 核心代码级鲁棒性修复 (Code-Level Patches)

在 `HF_HUB_OFFLINE=1` 严格断网及新版 `transformers` (5.x) 环境下，必须应用以下 Patch 以消除运行时崩溃。

### Patch 1: 动态绝对路径解析注入 (解决 `transformers` 离线寻址崩溃)

**目标文件**: `trellis2/pipelines/trellis2_image_to_3d.py` 及 `trellis2/pipelines/trellis2_texturing.py` **机制**: 拦截 `pipeline.json` 中的相对路径（如 `../../ckpts/...`），将其转换为基于当前执行目录的绝对路径，防止 `from_pretrained` 触发无效的云端缓存检测。

Python

```
# 在 super().from_pretrained() 解析 args 后拦截并转换
import os

# 示例修正逻辑：
model_rel_path = args['image_cond_model']['args']['model_name']
absolute_model_path = os.path.normpath(os.path.join(path, model_rel_path))
args['image_cond_model']['args']['model_name'] = absolute_model_path
```

### Patch 2: `transformers` 5.x 架构兼容性适配 (解决 DINOv3 属性缺失)

**目标文件**: `trellis2/modules/image_feature_extractor.py` (约 Line 86 处) **机制**: `transformers` 5.x 中 `DINOv3ViTModel` 的 Transformer Blocks 被移入额外的 `model` 层级。引入反射机制以兼容新旧版本 API 结构。

Python

```
# 替换原有的 for i, layer_module in enumerate(self.model.layer):
layers = self.model.model.layer if hasattr(self.model, 'model') and hasattr(self.model.model, 'layer') else self.model.layer
for i, layer_module in enumerate(layers):
    # 执行特征提取...
```

## 🔒 Step 5: 寻址固化与纯离线执行

修改配置文件以彻底斩断 `transformers` 库的云端探测逻辑。

**1. 修改流水线配置 (`ckpts/TRELLIS.2-4B/pipeline.json`)** 将原 Repo ID 修改为宿主机的**绝对路径**：

JSON

```
"image_cond_model": {
    "name": "DinoV3FeatureExtractor",
    "args": {
        "model_name": "/您的绝对路径/TRELLIS.2/ckpts/dinov3-vitl16-pretrain-lvd1689m"
    }
},
"rembg_model": {
    "name": "BiRefNet",
    "args": {
        "model_name": "/您的绝对路径/TRELLIS.2/ckpts/briaai"
    }
}
```

**2. 激活离线锁 (`example.py`)** 在入口脚本顶部解除环境隔离的注释：

Python

```
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
```

**3. 执行前向传播测试**

Bash

```
python example.py
```

## ⚠️ 已知 Issues 及复现陷阱 (Troubleshooting)

1. **`ValueError: numpy.dtype size changed`**:

   - **原因**: 宿主机全局缓存中的 `pandas` (依赖 NumPy 1.x) 渗透进了虚拟环境 (使用 NumPy 2.x)。
   - **解法**: 严格确保终端会话中 `export PYTHONNOUSERSITE=1` 处于激活状态。

2. **`fatal: 目标路径 '/tmp/extensions/...' 已经存在`**:

   - **原因**: 历史编译失败遗留的 C++ 源码占用了缓存。
   - **解法**: 执行 `. ./setup.sh` 前必须手动清空 `rm -rf /tmp/extensions/`。

   
