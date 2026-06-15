import os

# 1. 物理级斩断所有代理环境变量的残留污染，防止 httpx 引擎崩溃
# for k in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
#      if k in os.environ:
#        del os.environ[k]

# 2. 注入运行必须的环境变量
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'

import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 3. 严格校验并加载环境光贴图 (HDRI)
hdri_path = os.path.abspath('assets/hdri/forest.exr')
if not os.path.exists(hdri_path):
    raise FileNotFoundError(f"找不到环境光贴图: {hdri_path}")
hdri_img = cv2.imread(hdri_path, cv2.IMREAD_UNCHANGED)
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(hdri_img, cv2.COLOR_BGR2RGB),
    dtype=torch.float32, 
    device='cuda'
))

# 4. 强制使用绝对路径加载本地模型权重 (规避 '.' 开头被判定为非法 Repo ID 的 bug)
local_model_path = os.path.abspath("./ckpts/TRELLIS.2-4B")
print(f"从以下路径加载本地权重: {local_model_path}")

pipeline = Trellis2ImageTo3DPipeline.from_pretrained(local_model_path)
pipeline.cuda()
print("模型已成功加载到 4090D。")

# 5. 加载图像并执行前向传播
target_img_path = "assets/example_image/T.png"
print(f"正在处理图像: {target_img_path}")
image = Image.open(target_img_path)

mesh = pipeline.run(image)[0]
mesh.simplify(16777216) # nvdiffrast 光栅化内存限制

# 6. 准备输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 7. 渲染演示视频
print("正在渲染预览视频...")
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave(os.path.join(output_dir, "sample.mp4"), video, fps=15)

# 8. 导出结构化资产
print("正在导出 GLB 资产...")
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export(os.path.join(output_dir, "sample.glb"), extension_webp=True)
print(f"管道完成！资产已保存为 {os.path.join(output_dir, 'sample.glb')}")
