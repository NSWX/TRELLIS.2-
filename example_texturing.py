import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline

# 1. Load Pipeline
local_model_path = os.path.abspath("./ckpts/TRELLIS.2-4B")
print(f"从以下路径加载本地权重: {local_model_path}")
pipeline = Trellis2TexturingPipeline.from_pretrained(local_model_path, config_file="texturing_pipeline.json")
pipeline.cuda()

# 2. Load Mesh, image & Run
mesh = trimesh.load("assets/example_texturing/the_forgotten_knight.ply")
image = Image.open("assets/example_texturing/image.webp")
output = pipeline.run(mesh, image)

# 3. Render Mesh
output_dir = "output2"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "textured.glb")
output.export(output_path, extension_webp=True)
print(f"管道完成！纹理资产已保存为 {output_path}")