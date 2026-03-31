# Nvdiffrast 项目文档

## 1. 项目概述

**Nvdiffrast** 是由 NVIDIA 开发的一个高性能**可微分渲染**（Differentiable Rendering）PyTorch 库。它提供了基于光栅化的模块化渲染原语操作，可以无缝集成到 PyTorch 的自动微分（autograd）流程中，实现渲染过程的端到端梯度传播。

> 论文：*Modular Primitives for High-Performance Differentiable Rendering*  
> 作者：Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, Timo Aila  
> 论文链接：http://arxiv.org/abs/2011.03277

### 什么是可微分渲染？

传统的渲染流程（3D 场景 → 2D 图像）是不可微分的。可微分渲染使得梯度可以从 2D 图像反向传播到 3D 场景参数（顶点位置、颜色、纹理等），从而支持通过梯度下降进行 3D 重建、姿态估计、材质优化等任务。

---

## 2. 核心架构

项目由以下几部分组成：

| 目录 | 说明 |
|------|------|
| `csrc/common/` | CUDA/C++ 实现的核心渲染操作（光栅化、插值、纹理采样、抗锯齿） |
| `csrc/common/cudaraster/` | 自定义 CUDA 光栅化器实现 |
| `csrc/torch/` | PyTorch C++ 扩展绑定层 |
| `nvdiffrast/torch/` | Python API 层（`ops.py` 中封装所有操作） |
| `samples/torch/` | 示例代码 |

---

## 3. 四大核心操作

Nvdiffrast 提供四个模块化的可微分渲染原语，它们可以自由组合构建完整的渲染管线：

### 3.1 `rasterize()` — 光栅化

将 3D 三角形网格投影到 2D 图像平面，输出每个像素覆盖的三角形 ID 和重心坐标。

```python
rast_out, rast_db = dr.rasterize(glctx, pos_clip, tri, resolution=[height, width])
```

- **输入**：
  - `glctx`：光栅化上下文（`RasterizeCudaContext`）
  - `pos_clip`：裁剪空间顶点坐标，shape `[batch, num_vertices, 4]`（齐次坐标 `x, y, z, w`）
  - `tri`：三角形索引，shape `[num_triangles, 3]`，`torch.int32`
  - `resolution`：输出分辨率 `[height, width]`
- **输出**：
  - `rast_out`：shape `[batch, H, W, 4]`，包含 `(u, v, z/w, triangle_id)`
  - `rast_db`：shape `[batch, H, W, 4]`，重心坐标的图像空间导数

### 3.2 `interpolate()` — 属性插值

使用光栅化输出的重心坐标，对顶点属性（颜色、法线、纹理坐标等）进行插值。

```python
color, color_db = dr.interpolate(vertex_colors, rast_out, tri)
```

- **输入**：
  - `attr`：顶点属性，shape `[num_vertices, num_attrs]` 或 `[batch, num_vertices, num_attrs]`
  - `rast`：`rasterize()` 的主输出
  - `tri`：三角形索引
  - `rast_db`：（可选）用于计算属性的图像空间导数
  - `diff_attrs`：（可选）需要计算导数的属性索引列表
- **输出**：
  - 插值后的属性图像，shape `[batch, H, W, num_attrs]`
  - 属性的图像空间导数（如果请求）

### 3.3 `texture()` — 纹理采样

根据纹理坐标对纹理进行采样，支持 mipmap、多种滤波模式和边界模式。

```python
tex_out = dr.texture(texture_map, uv_coords, uv_da, filter_mode='linear-mipmap-linear')
```

- **滤波模式**：`nearest`、`linear`、`linear-mipmap-nearest`、`linear-mipmap-linear`、`auto`
- **边界模式**：`wrap`、`clamp`、`zero`、`cube`（立方体贴图）
- 支持预构建 mipmap（`texture_construct_mip()`）以提升重复调用的性能

### 3.4 `antialias()` — 抗锯齿

对渲染结果进行轮廓边缘抗锯齿处理，使得物体边界处的梯度可以正确传播。

```python
color_aa = dr.antialias(color, rast_out, pos_clip, tri)
```

这一步对于优化顶点位置（使物体的轮廓能产生正确梯度）至关重要。可通过 `antialias_construct_topology_hash()` 预计算拓扑信息以提升性能。

---

## 4. 安装

### 前置要求

- Python ≥ 3.6
- PyTorch（需 CUDA 支持）
- NVIDIA GPU + CUDA Toolkit
- `ninja`（用于 JIT 编译 C++/CUDA 扩展）

### pip 安装

```bash
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

> **注意**：必须使用 `--no-build-isolation` 标志，因为编译时需要访问已安装的 PyTorch。

### Docker 安装

```bash
docker build -t nvdiffrast .
```

基于 `nvcr.io/nvidia/pytorch` 镜像构建，会在容器内自动编译安装。

### 验证安装

```python
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()
print("nvdiffrast 安装成功!")
```

---

## 5. 快速上手

### 5.1 渲染一个彩色三角形

这是最简单的使用示例（对应 `samples/torch/triangle.py`）：

```python
import torch
import nvdiffrast.torch as dr
import numpy as np

# 1. 创建光栅化上下文
glctx = dr.RasterizeCudaContext()

# 2. 定义三角形顶点（裁剪空间坐标 x, y, z, w）
pos = torch.tensor([[[-0.8, -0.8, 0, 1],
                      [ 0.8, -0.8, 0, 1],
                      [-0.8,  0.8, 0, 1]]], dtype=torch.float32, device='cuda')

# 3. 定义顶点颜色 (RGB)
col = torch.tensor([[[1, 0, 0],    # 红
                      [0, 1, 0],    # 绿
                      [0, 0, 1]]], dtype=torch.float32, device='cuda')  # 蓝

# 4. 定义三角形索引
tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')

# 5. 光栅化
rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256, 256])

# 6. 插值顶点颜色
out, _ = dr.interpolate(col, rast, tri)

# 7. 转为图像
img = out.cpu().numpy()[0, ::-1, :, :]  # 垂直翻转
img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
```

### 5.2 可微分渲染优化（以 cube 示例说明）

`samples/torch/cube.py` 展示了如何通过梯度下降优化 3D 模型参数：

```python
import torch
import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()

# 加载目标网格和初始网格
# ... 省略数据加载 ...

# 将顶点颜色设为可优化参数
vtx_col_opt = torch.tensor(initial_colors, dtype=torch.float32, device='cuda', requires_grad=True)
optimizer = torch.optim.Adam([vtx_col_opt], lr=1e-2)

for i in range(num_iterations):
    optimizer.zero_grad()

    # 前向渲染
    pos_clip = transform_pos(mvp_matrix, vtx_pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[res, res])
    color, _ = dr.interpolate(vtx_col_opt[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)

    # 计算损失（与目标图像的差异）
    loss = torch.mean((color - target_image) ** 2)

    # 反向传播 + 优化
    loss.backward()
    optimizer.step()
```

关键点：由于所有四个操作都支持自动微分，梯度可以从图像损失一路传播到顶点位置和颜色。

---

## 6. 两种批处理模式

### Instanced 模式

每个 batch 元素使用独立的顶点位置，`pos` shape 为 `[batch, num_vertices, 4]`：

```python
# batch=4，每个元素有独立的顶点位置
pos = torch.randn(4, 100, 4, device='cuda')
rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256, 256])
```

### Range 模式

所有 batch 元素共享顶点数据，但通过 `ranges` 指定各自使用的三角形范围。`pos` shape 为 `[num_vertices, 4]`（2D）：

```python
# ranges[i] = [start_idx, count]，指定第 i 个 batch 使用 tri 中的哪些三角形
ranges = torch.tensor([[0, 100], [100, 50]], dtype=torch.int32)
rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256, 256], ranges=ranges)
```

---

## 7. 高级功能

### 深度剥离（Depth Peeling）

通过 `DepthPeeler` 上下文管理器逐层渲染多个深度层，用于透明物体或多层分析：

```python
with dr.DepthPeeler(glctx, pos, tri, resolution=[256, 256]) as peeler:
    rast_first, db_first = peeler.rasterize_next_layer()   # 最近的表面
    rast_second, db_second = peeler.rasterize_next_layer()  # 第二层表面
```

### Mipmap 预构建

当纹理不变时，预构建 mipmap 避免重复计算：

```python
mip = dr.texture_construct_mip(tex)
for uv in uv_list:
    result = dr.texture(tex, uv, mip=mip, filter_mode='linear-mipmap-linear')
```

### 拓扑 Hash 预构建

当三角形拓扑不变时，预构建拓扑 hash 提升抗锯齿性能：

```python
topo_hash = dr.antialias_construct_topology_hash(tri)
for frame in frames:
    color_aa = dr.antialias(color, rast, pos, tri, topology_hash=topo_hash)
```

---

## 8. 示例程序

| 示例 | 文件 | 说明 |
|------|------|------|
| Triangle | `samples/torch/triangle.py` | 最简示例：渲染一个彩色三角形 |
| Cube | `samples/torch/cube.py` | 通过梯度下降优化立方体顶点颜色 |
| Earth | `samples/torch/earth.py` | 地球纹理渲染和姿态优化 |
| Pose | `samples/torch/pose.py` | 3D 物体姿态估计 |
| Envphong | `samples/torch/envphong.py` | 环境光照 + Phong 着色优化 |

运行示例：

```bash
cd samples/torch
python triangle.py
python cube.py --outdir results/cube
```

---

## 9. API 速查

```python
import nvdiffrast.torch as dr

# 上下文
glctx = dr.RasterizeCudaContext(device='cuda:0')

# 核心操作
rast, rast_db = dr.rasterize(glctx, pos, tri, resolution)
attr, attr_db = dr.interpolate(attr, rast, tri, rast_db=rast_db, diff_attrs='all')
tex_out       = dr.texture(tex, uv, uv_da=uv_da, filter_mode='auto', boundary_mode='wrap')
color_aa      = dr.antialias(color, rast, pos, tri)

# 预构建辅助
mip       = dr.texture_construct_mip(tex)
topo_hash = dr.antialias_construct_topology_hash(tri)

# 深度剥离
with dr.DepthPeeler(glctx, pos, tri, resolution) as peeler:
    rast_layer, db_layer = peeler.rasterize_next_layer()

# 日志
dr.set_log_level(0)  # 0=Info, 1=Warning, 2=Error, 3=Fatal
```

---

## 10. 典型渲染管线流程

```
顶点位置 (pos)          顶点属性 (color/uv/normal)       纹理 (tex)
      │                          │                           │
      ▼                          │                           │
 ┌──────────┐                    │                           │
 │ rasterize │ ──→ rast, rast_db │                           │
 └──────────┘         │          │                           │
                      ▼          ▼                           │
               ┌─────────────┐                               │
               │ interpolate │ ──→ uv, uv_da ───┐           │
               └─────────────┘                    │           │
                      │                           ▼           ▼
                      │                    ┌───────────┐
                      │                    │  texture   │
                      │                    └───────────┘
                      │                           │
                      ▼                           ▼
               color (直接插值)         color (纹理采样)
                      │                           │
                      └───────────┬───────────────┘
                                  ▼
                          ┌─────────────┐
                          │  antialias  │
                          └─────────────┘
                                  │
                                  ▼
                            最终渲染图像
```

---

## 11. 注意事项

1. **所有输入张量必须连续**（contiguous）且位于 GPU 内存中（`ranges` 参数除外，需在 CPU 上）。
2. **顶点索引一致性**：同一个顶点在不同三角形中必须使用相同的索引，否则 `antialias()` 会错误地将共享边识别为轮廓边。
3. **坐标系统**：`rasterize()` 输入需要裁剪空间坐标（clip space），即经过 MVP 矩阵变换后的齐次坐标 `(x, y, z, w)`。
4. **`--no-build-isolation`**：安装时必须使用此标志，否则编译扩展时找不到 PyTorch。
5. 版本：当前版本为 **0.4.0**。
