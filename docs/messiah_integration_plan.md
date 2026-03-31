# nvdiffrast × Messiah Engine 集成方案

## 一、总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                  NvDiffRast Optimizer Editor                 │
│                     (独立 Python/Qt 应用)                     │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ 场景导入  │  │ 可微渲染管线  │  │  优化控制面板       │    │
│  │ (Mesh/Mat │  │ (模拟Messiah │  │  (Loss/LR/Iter)    │    │
│  │  /Tex)    │  │  HybridPipe) │  │                    │    │
│  └────┬─────┘  └──────┬───────┘  └────────┬───────────┘    │
│       │               │                    │                │
│  ┌────▼───────────────▼────────────────────▼───────────┐    │
│  │              nvdiffrast 可微渲染核心                  │    │
│  │  rasterize → interpolate → texture → shade → AA     │    │
│  │              ↕ PyTorch autograd 反向传播              │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │              优化结果导出                             │    │
│  │  优化后贴图(.dds/.png) | 优化后材质参数(.json/.xml) │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │ 文件交换 / Socket通信
┌──────────────────────▼──────────────────────────────────────┐
│                    Messiah Engine Editor                      │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ 场景导出  │  │ 对比预览      │  │  导入优化结果       │    │
│  │ Plugin    │  │ (Before/After)│  │  Plugin             │    │
│  └──────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、核心思路

nvdiffrast 是一个**可微分光栅化**库，能够将2D图像的像素差异(Loss)反向传播到3D场景参数（顶点位置、贴图像素、材质参数）。

**核心价值**：给定一个 Reference 渲染目标（高质量 ground truth），通过梯度下降自动优化贴图/Shader参数，使低成本渲染结果尽可能逼近 Reference。

### 适用场景

| 场景 | 输入 | 优化目标 | 输出 |
|------|------|---------|------|
| **贴图压缩优化** | 原始高精度贴图 + 场景 | 压缩后的贴图像素 | 优化的低分辨率/压缩贴图 |
| **LOD贴图烘焙** | 高模渲染结果 | 低模的贴图 | 低模上使用的优化贴图 |
| **Shader简化** | 复杂Shader渲染结果 | 简化Shader的参数 | 等效简单Shader参数集 |
| **材质参数拟合** | 参考照片/渲染图 | PBR材质参数 | BaseColor/Roughness/Metallic等 |
| **法线贴图优化** | 高模几何细节 | 法线贴图像素 | 优化的法线贴图 |
| **光照贴图优化** | 全局光照参考 | Lightmap像素 | 优化的光照贴图 |

---

## 三、可微渲染管线设计（模拟 Messiah HybridPipeline）

### 3.1 Messiah 渲染管线核心特征

根据引擎代码分析，Messiah 的 HybridPipeline 核心特征为：

- **GBuffer Layout**：5-RT Thin GBuffer（Emission, VirtualLighting, Roughness/Custom, Normal/ShadingModelID, BaseColor）
- **BRDF**：GGX 分布 + Schlick_disney 几何项 + Schlick Fresnel + Lambert/Burley 漫反射
- **Shading Models**：19种（DefaultLit, SSS, Hair, Cloth, Eye, Foliage 等）
- **后处理**：TSAA, Bloom, SSR, SSAO/GTAO, VolumetricFog 等

### 3.2 PyTorch 可微管线实现

```python
# messiah_pipeline.py - 用 nvdiffrast + PyTorch 模拟 Messiah 渲染管线

import torch
import nvdiffrast.torch as dr
import torch.nn.functional as F
import math

class MessiahDiffPipeline:
    """模拟 Messiah HybridPipeline 的可微分渲染管线"""

    def __init__(self, resolution=(1024, 1024), device='cuda'):
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.resolution = resolution
        self.device = device

    # ========== GGX BRDF (匹配 Messiah BRDF.fxh) ==========

    @staticmethod
    def ggx_distribution(NdotH, roughness):
        """GGX/Trowbridge-Reitz NDF - 对应 Messiah D_GGX()"""
        a2 = roughness * roughness
        a4 = a2 * a2
        d = NdotH * NdotH * (a4 - 1.0) + 1.0
        return a4 / (math.pi * d * d + 1e-7)

    @staticmethod
    def schlick_fresnel(F0, VdotH):
        """Schlick Fresnel - 对应 Messiah F_Schlick()"""
        return F0 + (1.0 - F0) * torch.pow(1.0 - VdotH, 5.0)

    @staticmethod
    def smith_ggx_geometry(NdotV, NdotL, roughness):
        """Smith GGX 几何遮蔽 - 对应 Messiah Vis_Schlick_Disney()"""
        k = (roughness + 1.0) ** 2 / 8.0
        g1_v = NdotV / (NdotV * (1.0 - k) + k + 1e-7)
        g1_l = NdotL / (NdotL * (1.0 - k) + k + 1e-7)
        return g1_v * g1_l

    def pbr_shading(self, normal, view_dir, light_dir, light_color,
                    base_color, metallic, roughness, ao=None):
        """
        PBR 着色 - 匹配 Messiah DefaultLit Shading Model
        对应 Deferred.fx 中的 DefaultLit 分支
        """
        # 确保 roughness 最小值 (对应 MIN_ROUGHNESS = 0.08)
        roughness = torch.clamp(roughness, min=0.08)

        H = F.normalize(view_dir + light_dir, dim=-1)
        NdotL = torch.clamp(torch.sum(normal * light_dir, dim=-1, keepdim=True), 0.0, 1.0)
        NdotV = torch.clamp(torch.sum(normal * view_dir, dim=-1, keepdim=True), 0.0, 1.0)
        NdotH = torch.clamp(torch.sum(normal * H, dim=-1, keepdim=True), 0.0, 1.0)
        VdotH = torch.clamp(torch.sum(view_dir * H, dim=-1, keepdim=True), 0.0, 1.0)

        # Dielectric F0 = 0.04, metallic blends toward base_color
        F0 = 0.04 * (1.0 - metallic) + base_color * metallic

        # BRDF components
        D = self.ggx_distribution(NdotH, roughness)
        G = self.smith_ggx_geometry(NdotV, NdotL, roughness)
        Fr = self.schlick_fresnel(F0, VdotH)

        # Specular BRDF
        specular = (D * G * Fr) / (4.0 * NdotV * NdotL + 1e-7)

        # Diffuse (Lambert) - 对应 Messiah Diff_Lambert()
        kD = (1.0 - Fr) * (1.0 - metallic)
        diffuse = kD * base_color / math.pi

        # Final color
        color = (diffuse + specular) * light_color * NdotL

        if ao is not None:
            color = color * ao

        return color

    def render(self, pos_clip, tri, vtx_attr, textures, material_params,
               camera_pos, light_dir, light_color):
        """
        完整的可微分前向渲染

        Args:
            pos_clip:        [B, V, 4] clip-space 顶点位置
            tri:             [T, 3] 三角形索引
            vtx_attr:        dict { 'normal': [V,3], 'uv': [V,2], 'tangent': [V,4] }
            textures:        dict { 'base_color': [1,H,W,3], 'normal': [1,H,W,3],
                                    'roughness': [1,H,W,1], 'metallic': [1,H,W,1] }
            material_params: dict { 可选的额外材质参数 }
            camera_pos:      [B, 3] 相机位置(world space)
            light_dir:       [3] 方向光方向
            light_color:     [3] 光颜色/强度
        """
        B = pos_clip.shape[0]

        # === 1. Rasterize (光栅化) ===
        rast_out, rast_db = dr.rasterize(
            self.glctx, pos_clip, tri, resolution=self.resolution
        )

        # === 2. Interpolate (属性插值) ===
        # 插值法线
        normal_interp, _ = dr.interpolate(
            vtx_attr['normal'][None, ...], rast_out, tri
        )
        normal_interp = F.normalize(normal_interp, dim=-1)

        # 插值UV坐标 (带导数，用于 mipmapping)
        uv_interp, uv_da = dr.interpolate(
            vtx_attr['uv'][None, ...], rast_out, tri,
            rast_db=rast_db, diff_attrs='all'
        )

        # 插值世界坐标 (用于计算 view direction)
        if 'pos_world' in vtx_attr:
            pos_world, _ = dr.interpolate(
                vtx_attr['pos_world'][None, ...], rast_out, tri
            )

        # === 3. Texture Sampling (纹理采样, 模拟 Messiah 纹理系统) ===
        base_color = dr.texture(
            textures['base_color'], uv_interp, uv_da,
            filter_mode='linear-mipmap-linear'
        )

        # 法线贴图 (tangent space → world space)
        if 'normal' in textures:
            normal_map = dr.texture(
                textures['normal'], uv_interp, uv_da,
                filter_mode='linear-mipmap-linear'
            )
            normal_map = normal_map * 2.0 - 1.0
            # TODO: TBN矩阵变换 (需要 tangent/bitangent)
            # 简化版：直接用插值法线 + 法线贴图扰动
            normal_interp = F.normalize(normal_interp + normal_map[..., :3] * 0.5, dim=-1)

        roughness_tex = dr.texture(
            textures['roughness'], uv_interp, uv_da,
            filter_mode='linear-mipmap-linear'
        )

        metallic_tex = dr.texture(
            textures['metallic'], uv_interp, uv_da,
            filter_mode='linear-mipmap-linear'
        )

        # === 4. PBR Shading (匹配 Messiah DefaultLit) ===
        view_dir = F.normalize(
            camera_pos[:, None, None, :] - pos_world, dim=-1
        )
        L = F.normalize(light_dir.unsqueeze(0).unsqueeze(0).unsqueeze(0), dim=-1)

        color = self.pbr_shading(
            normal=normal_interp,
            view_dir=view_dir,
            light_dir=L.expand_as(view_dir),
            light_color=light_color,
            base_color=base_color[..., :3],
            metallic=metallic_tex[..., :1],
            roughness=roughness_tex[..., :1]
        )

        # === 5. Antialias (抗锯齿 + 梯度修正) ===
        color = dr.antialias(color, rast_out, pos_clip, tri)

        # 背景mask (未覆盖像素)
        mask = (rast_out[..., 3:4] > 0).float()
        color = color * mask

        return color, mask, rast_out

```

---

## 四、外部编辑器工具设计

### 4.1 技术栈

| 组件 | 技术选型 | 理由 |
|------|---------|------|
| GUI 框架 | **PyQt6** | 与 Messiah 编辑器一致(同为Qt), 可复用组件 |
| 可微渲染 | **nvdiffrast + PyTorch** | GPU加速可微光栅化 |
| 3D预览 | **PyOpenGL / Qt3DWidget** | 实时预览优化过程 |
| 通信协议 | **JSON-RPC over TCP/Named Pipe** | 与 Messiah 编辑器进程通信 |
| 资源格式 | **glTF/FBX 导入, DDS/PNG 导出** | 兼容 Messiah 资源管线 |

### 4.2 编辑器功能模块

```
NvDiffRast Optimizer Editor
├── 📁 Scene Panel (场景面板)
│   ├── 从 Messiah 导入场景 (Mesh + Material + Camera + Light)
│   ├── 3D 视口预览 (可旋转/缩放)
│   └── 场景元素树 (选择要优化的物体)
│
├── 🎯 Reference Panel (参考图面板)
│   ├── 从 Messiah 截取高质量渲染参考图
│   ├── 手动导入参考图
│   └── 多视角参考图管理
│
├── ⚙️ Optimization Panel (优化面板)
│   ├── 优化模式选择:
│   │   ├── Texture Optimization (贴图优化)
│   │   ├── Material Fitting (材质参数拟合)
│   │   ├── Shader Simplification (Shader简化)
│   │   └── Normal Map Baking (法线贴图烘焙)
│   ├── 优化参数:
│   │   ├── Learning Rate / Schedule
│   │   ├── Loss Function (L1/L2/Perceptual/SSIM)
│   │   ├── Regularization (Smoothness/Sparsity)
│   │   └── Iteration Count
│   └── 约束条件:
│       ├── 贴图分辨率约束
│       ├── 参数范围约束
│       └── 内存预算约束
│
├── 📊 Monitor Panel (监控面板)
│   ├── Loss 曲线实时绘制
│   ├── 当前渲染 vs 参考图对比 (Side by Side / Overlay / Diff)
│   ├── PSNR / SSIM 指标
│   └── GPU 内存/时间统计
│
└── 📤 Export Panel (导出面板)
    ├── 导出优化后的贴图 (.dds / .png / .tga)
    ├── 导出材质参数 (.json / Messiah .xml)
    ├── 生成优化报告
    └── 一键回写 Messiah 项目
```

### 4.3 编辑器 UI 框架代码

```python
# editor/main_window.py

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QDockWidget,
                              QWidget, QVBoxLayout, QSplitter)
from PyQt6.QtCore import Qt

class OptimizerMainWindow(QMainWindow):
    """NvDiffRast Optimizer 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NvDiffRast Shader/Texture Optimizer for Messiah")
        self.setMinimumSize(1600, 900)
        self._setup_ui()

    def _setup_ui(self):
        # 中央: 3D视口 + 对比视图
        central = QSplitter(Qt.Orientation.Horizontal)
        central.addWidget(self._create_viewport_widget())    # nvdiffrast实时渲染
        central.addWidget(self._create_reference_widget())   # 参考图/对比
        self.setCentralWidget(central)

        # 左侧: 场景树
        self._add_dock("Scene", self._create_scene_panel(),
                       Qt.DockWidgetArea.LeftDockWidgetArea)

        # 右侧: 优化控制
        self._add_dock("Optimization", self._create_optim_panel(),
                       Qt.DockWidgetArea.RightDockWidgetArea)

        # 底部: Loss曲线 + 日志
        self._add_dock("Monitor", self._create_monitor_panel(),
                       Qt.DockWidgetArea.BottomDockWidgetArea)

    def _add_dock(self, title, widget, area):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)

    # ... 各面板创建方法
```

---

## 五、优化工作流详解

### 5.1 贴图优化流程

```
目标: 从高分辨率贴图生成优化的低分辨率贴图，保持视觉质量

Step 1: 导入
  ├─ 从 Messiah 导出: Mesh(.fbx) + 高精度贴图(BaseColor/Normal/Roughness/Metallic)
  └─ 设置相机阵列(多视角), 匹配 Messiah 相机参数

Step 2: 生成 Reference
  ├─ 使用 MessiahDiffPipeline + 原始高精贴图渲染多视角参考图
  └─ 或从 Messiah 直接截取渲染结果

Step 3: 初始化优化目标
  ├─ 创建低分辨率贴图 (可学习的 torch.nn.Parameter)
  │   base_color_opt = torch.nn.Parameter(
  │       F.interpolate(base_color_hires, size=(256,256))
  │   )
  └─ 设置约束: 值域[0,1], 平滑度正则化

Step 4: 优化循环
  for each iteration:
    ├─ 随机/遍历 相机视角
    ├─ 用低分辨率贴图渲染 → color_opt
    ├─ 计算 Loss:
    │   loss = L2(color_opt, color_ref)
    │        + λ_perceptual * PerceptualLoss(color_opt, color_ref)
    │        + λ_smooth * SmoothnessRegularizer(texture_opt)
    ├─ loss.backward()  → 梯度回传到贴图像素
    └─ optimizer.step() → 更新贴图

Step 5: 导出
  ├─ 保存优化后的贴图为 DDS/PNG
  └─ 导入回 Messiah
```

#### 核心优化代码

```python
# optimizer/texture_optimizer.py

class TextureOptimizer:
    """贴图优化器 - 通过可微渲染优化贴图"""

    def __init__(self, pipeline, mesh_data, cameras, reference_images,
                 target_resolution=(256, 256)):
        self.pipeline = pipeline
        self.mesh = mesh_data
        self.cameras = cameras  # List of camera matrices
        self.refs = reference_images  # [N_views, H, W, 3]

        # 可优化的贴图参数
        init_tex = F.interpolate(
            mesh_data['base_color_hires'].permute(0,3,1,2),
            size=target_resolution,
            mode='bilinear'
        ).permute(0,2,3,1)

        self.base_color_opt = torch.nn.Parameter(init_tex.contiguous())
        self.optimizer = torch.optim.Adam([self.base_color_opt], lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000
        )

    def step(self, view_idx=None):
        """执行一步优化"""
        if view_idx is None:
            view_idx = torch.randint(len(self.cameras), (1,)).item()

        mvp = self.cameras[view_idx]
        ref = self.refs[view_idx]

        # 前向渲染
        textures = {
            'base_color': torch.sigmoid(self.base_color_opt),  # 约束到[0,1]
            'roughness': self.mesh['roughness_tex'],
            'metallic': self.mesh['metallic_tex'],
            'normal': self.mesh['normal_tex'],
        }

        pos_clip = transform_vertices(self.mesh['vertices'], mvp)
        color_opt, mask, _ = self.pipeline.render(
            pos_clip, self.mesh['triangles'],
            self.mesh['vtx_attr'], textures,
            {}, self.cameras[view_idx]['pos'],
            self.mesh['light_dir'], self.mesh['light_color']
        )

        # 计算Loss
        loss = self._compute_loss(color_opt, ref, mask)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), color_opt.detach()

    def _compute_loss(self, rendered, reference, mask):
        """多项损失函数"""
        # L2 像素损失
        l2 = torch.mean((rendered - reference) ** 2 * mask)

        # 平滑度正则化 (避免噪声贴图)
        tex = torch.sigmoid(self.base_color_opt)
        dx = tex[:, 1:, :, :] - tex[:, :-1, :, :]
        dy = tex[:, :, 1:, :] - tex[:, :, :-1, :]
        smoothness = torch.mean(dx**2) + torch.mean(dy**2)

        return l2 + 0.001 * smoothness

    def export(self, path):
        """导出优化后的贴图"""
        tex = torch.sigmoid(self.base_color_opt).detach().cpu()
        # 转换为 uint8 并保存
        tex_np = (tex[0].numpy() * 255).clip(0, 255).astype('uint8')
        # 保存为 PNG/DDS
        save_texture(tex_np, path)
```

### 5.2 Shader 参数优化/简化流程

```
目标: 将复杂Shader(如SSS/Cloth)的视觉效果拟合到简单Shader(如DefaultLit)

Step 1: 在 Messiah 中用复杂Shader渲染多视角参考图 (Ground Truth)

Step 2: 在优化器中构建简化Shader的可微版本
  ├─ 定义可学习参数: base_color, roughness, metallic, emission 等
  └─ 使用 MessiahDiffPipeline.pbr_shading() 作为简化Shader

Step 3: 优化循环
  for each iteration:
    ├─ 用简化Shader渲染
    ├─ 与复杂Shader参考图计算Loss
    ├─ 反向传播更新材质参数
    └─ 可选: 同时优化贴图和标量参数

Step 4: 导出简化后的材质参数集
```

#### Shader 简化优化代码

```python
# optimizer/shader_simplifier.py

class ShaderSimplifier:
    """Shader简化器 - 将复杂Shader效果拟合为简单PBR参数"""

    def __init__(self, pipeline, mesh_data, cameras, reference_images):
        self.pipeline = pipeline

        # 可优化的标量材质参数
        self.roughness = torch.nn.Parameter(torch.tensor([0.5], device='cuda'))
        self.metallic = torch.nn.Parameter(torch.tensor([0.0], device='cuda'))
        self.emission_strength = torch.nn.Parameter(torch.tensor([0.0], device='cuda'))

        # 可优化的贴图参数 (如果需要调整贴图)
        self.base_color_bias = torch.nn.Parameter(
            torch.zeros(1, 1, 1, 3, device='cuda')
        )

        self.optimizer = torch.optim.Adam([
            {'params': [self.roughness, self.metallic], 'lr': 1e-2},
            {'params': [self.emission_strength], 'lr': 1e-3},
            {'params': [self.base_color_bias], 'lr': 5e-3},
        ])

    def get_optimized_params(self):
        """获取优化后的材质参数 (可导出为 Messiah XML)"""
        return {
            'roughness': torch.sigmoid(self.roughness).item(),
            'metallic': torch.sigmoid(self.metallic).item(),
            'emission_strength': torch.relu(self.emission_strength).item(),
            'base_color_bias': self.base_color_bias.detach().cpu().numpy().tolist(),
        }

    def export_messiah_material(self, path):
        """导出为 Messiah 材质 XML 格式"""
        params = self.get_optimized_params()
        xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<Material>
    <ShadingModel>DefaultLit</ShadingModel>
    <Parameters>
        <Roughness>{params['roughness']:.4f}</Roughness>
        <Metallic>{params['metallic']:.4f}</Metallic>
        <EmissionStrength>{params['emission_strength']:.4f}</EmissionStrength>
    </Parameters>
</Material>"""
        with open(path, 'w') as f:
            f.write(xml_content)
```

---

## 六、与 Messiah 引擎的集成接口

### 6.1 场景导出插件 (Messiah 侧)

在 Messiah 编辑器中添加 Python 插件，将当前场景数据导出为优化器可读格式：

```python
# messiah_plugin/export_for_optimizer.py
# 运行在 Messiah Editor 的 Python 环境中 (Editor/QtScript/)

import json
import struct

def export_scene_for_optimizer(output_dir):
    """从 Messiah 编辑器导出当前场景数据"""

    # 通过 MEditor API 获取选中物体信息
    selected = MEditor.GetSelectedObjects()

    scene_data = {
        'meshes': [],
        'materials': [],
        'cameras': [],
        'lights': [],
    }

    for obj in selected:
        # 导出 Mesh
        mesh = obj.GetMesh()
        mesh_data = {
            'vertices': mesh.GetVertices().tolist(),    # [V, 3]
            'normals': mesh.GetNormals().tolist(),      # [V, 3]
            'uvs': mesh.GetUVs(0).tolist(),             # [V, 2]
            'indices': mesh.GetIndices().tolist(),       # [T, 3]
        }
        save_binary_mesh(f"{output_dir}/{obj.name}.mesh", mesh_data)

        # 导出材质
        mat = obj.GetMaterial()
        mat_data = {
            'shading_model': mat.GetShadingModel(),
            'base_color_tex': mat.GetTexturePath('BaseColor'),
            'normal_tex': mat.GetTexturePath('Normal'),
            'roughness_tex': mat.GetTexturePath('Roughness'),
            'metallic_tex': mat.GetTexturePath('Metallic'),
            'scalar_params': {
                'roughness': mat.GetScalar('Roughness'),
                'metallic': mat.GetScalar('Metallic'),
            }
        }
        scene_data['materials'].append(mat_data)

    # 导出当前相机
    cam = MEditor.GetActiveCamera()
    scene_data['cameras'].append({
        'view_matrix': cam.GetViewMatrix().tolist(),
        'proj_matrix': cam.GetProjectionMatrix().tolist(),
        'position': cam.GetPosition().tolist(),
        'fov': cam.GetFOV(),
        'near': cam.GetNearPlane(),
        'far': cam.GetFarPlane(),
    })

    # 导出灯光
    for light in MEditor.GetLights():
        scene_data['lights'].append({
            'type': light.GetType(),  # directional/point/spot
            'direction': light.GetDirection().tolist(),
            'color': light.GetColor().tolist(),
            'intensity': light.GetIntensity(),
        })

    # 保存场景描述
    with open(f"{output_dir}/scene.json", 'w') as f:
        json.dump(scene_data, f, indent=2)

    # 导出参考渲染图
    MEditor.CaptureFramebuffer(f"{output_dir}/reference.png")

    return scene_data
```

### 6.2 实时通信桥接

```python
# bridge/messiah_bridge.py
# 优化器与 Messiah 编辑器之间的通信桥

import socket
import json
import threading

class MessiahBridge:
    """
    通过 Named Pipe / TCP 与 Messiah 编辑器通信
    支持:
    - pull: 从 Messiah 拉取场景/渲染数据
    - push: 将优化结果推回 Messiah
    - sync: 实时同步相机/光照变化
    """

    def __init__(self, host='127.0.0.1', port=9527):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def send_command(self, cmd, params=None):
        """发送 JSON-RPC 命令到 Messiah"""
        request = {
            'jsonrpc': '2.0',
            'method': cmd,
            'params': params or {},
            'id': 1,
        }
        data = json.dumps(request).encode('utf-8')
        self.sock.sendall(len(data).to_bytes(4, 'little') + data)

        # 接收响应
        size = int.from_bytes(self.sock.recv(4), 'little')
        response = json.loads(self.sock.recv(size).decode('utf-8'))
        return response.get('result')

    # === 常用命令 ===

    def pull_scene(self, output_dir):
        """从 Messiah 拉取当前场景"""
        return self.send_command('export_scene', {'output_dir': output_dir})

    def pull_reference(self, resolution, camera_params):
        """从 Messiah 获取参考渲染图"""
        return self.send_command('capture_frame', {
            'resolution': resolution,
            'camera': camera_params,
        })

    def push_texture(self, texture_path, asset_path):
        """将优化后的贴图推送回 Messiah"""
        return self.send_command('import_texture', {
            'source': texture_path,
            'target': asset_path,
        })

    def push_material(self, material_data, material_name):
        """将优化后的材质参数推送回 Messiah"""
        return self.send_command('update_material', {
            'name': material_name,
            'params': material_data,
        })

    def trigger_hot_reload(self):
        """触发 Messiah Shader/贴图热重载 (利用 ShaderWatcher)"""
        return self.send_command('hot_reload', {})
```

### 6.3 Messiah 侧的通信服务端

```python
# messiah_plugin/optimizer_server.py
# 在 Messiah Editor Python 环境中运行的 JSON-RPC 服务

import threading
import socket
import json

class OptimizerServer:
    """运行在 Messiah Editor 内的 JSON-RPC 服务端"""

    def __init__(self, port=9527):
        self.port = port
        self.handlers = {
            'export_scene': self._handle_export_scene,
            'capture_frame': self._handle_capture_frame,
            'import_texture': self._handle_import_texture,
            'update_material': self._handle_update_material,
            'hot_reload': self._handle_hot_reload,
        }

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('127.0.0.1', self.port))
        self.server.listen(1)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while True:
            conn, addr = self.server.accept()
            threading.Thread(
                target=self._handle_client, args=(conn,), daemon=True
            ).start()

    def _handle_client(self, conn):
        try:
            while True:
                size = int.from_bytes(conn.recv(4), 'little')
                data = json.loads(conn.recv(size).decode('utf-8'))
                handler = self.handlers.get(data['method'])
                if handler:
                    result = handler(data.get('params', {}))
                else:
                    result = {'error': f"Unknown method: {data['method']}"}
                resp = json.dumps({'jsonrpc': '2.0', 'result': result, 'id': data['id']})
                resp_bytes = resp.encode('utf-8')
                conn.sendall(len(resp_bytes).to_bytes(4, 'little') + resp_bytes)
        finally:
            conn.close()

    def _handle_hot_reload(self, params):
        """触发Shader/贴图热重载"""
        MEngine.AddCallback(0, lambda: MEditor.EvalOnEditorPython(
            'MEditor.ReloadShaders(); MEditor.ReloadTextures()'
        ))
        return {'status': 'ok'}

    # ... 其他 handler 实现
```

---

## 七、Loss 函数设计

匹配 Messiah 渲染质量需要多种 Loss 函数组合：

```python
# optimizer/losses.py

import torch
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(torch.nn.Module):
    """VGG-based Perceptual Loss - 感知质量比纯L2更接近人眼"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.cuda()

    def forward(self, x, y):
        # x, y: [B, H, W, 3] → [B, 3, H, W]
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        return F.mse_loss(self.vgg(x), self.vgg(y))

class SSIMLoss(torch.nn.Module):
    """SSIM Loss - 结构相似性，对Shader效果差异敏感"""
    def forward(self, x, y, window_size=11):
        C1, C2 = 0.01**2, 0.03**2
        mu_x = F.avg_pool2d(x.permute(0,3,1,2), window_size, 1, window_size//2)
        mu_y = F.avg_pool2d(y.permute(0,3,1,2), window_size, 1, window_size//2)
        sigma_x2 = F.avg_pool2d(x.permute(0,3,1,2)**2, window_size, 1, window_size//2) - mu_x**2
        sigma_y2 = F.avg_pool2d(y.permute(0,3,1,2)**2, window_size, 1, window_size//2) - mu_y**2
        sigma_xy = F.avg_pool2d(
            x.permute(0,3,1,2)*y.permute(0,3,1,2), window_size, 1, window_size//2
        ) - mu_x*mu_y
        ssim = ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sigma_x2+sigma_y2+C2))
        return 1.0 - ssim.mean()

class CompositeLoss(torch.nn.Module):
    """组合Loss - 用于Messiah贴图/Shader优化"""
    def __init__(self, w_l2=1.0, w_perceptual=0.1, w_ssim=0.05):
        super().__init__()
        self.w_l2 = w_l2
        self.w_perceptual = w_perceptual
        self.w_ssim = w_ssim
        self.perceptual = PerceptualLoss()
        self.ssim = SSIMLoss()

    def forward(self, rendered, reference, mask=None):
        if mask is not None:
            rendered = rendered * mask
            reference = reference * mask
        loss = self.w_l2 * F.mse_loss(rendered, reference)
        loss += self.w_perceptual * self.perceptual(rendered, reference)
        loss += self.w_ssim * self.ssim(rendered, reference)
        return loss
```

---

## 八、项目文件结构

```
nvdiffrast-messiah-optimizer/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── default_config.yaml          # 默认优化配置
│   └── messiah_brdf_params.yaml     # Messiah BRDF 参数映射表
│
├── editor/                          # 外部编辑器 (PyQt6)
│   ├── main.py                      # 入口
│   ├── main_window.py               # 主窗口
│   ├── panels/
│   │   ├── scene_panel.py           # 场景导入/浏览
│   │   ├── reference_panel.py       # 参考图管理
│   │   ├── optim_panel.py           # 优化参数控制
│   │   ├── monitor_panel.py         # Loss曲线/对比视图
│   │   └── export_panel.py          # 结果导出
│   └── widgets/
│       ├── viewport_3d.py           # 3D预览视口
│       ├── image_compare.py         # Before/After对比
│       └── loss_chart.py            # Loss曲线图
│
├── pipeline/                        # 可微渲染管线
│   ├── messiah_pipeline.py          # Messiah HybridPipeline 可微版
│   ├── brdf.py                      # GGX BRDF (匹配 BRDF.fxh)
│   ├── shading_models.py            # 各 ShadingModel 实现
│   ├── tonemapping.py               # 色调映射 (匹配 Messiah 后处理)
│   └── camera.py                    # 相机/投影矩阵工具
│
├── optimizer/                       # 优化器
│   ├── texture_optimizer.py         # 贴图优化
│   ├── material_fitter.py           # 材质参数拟合
│   ├── shader_simplifier.py         # Shader简化
│   ├── normal_baker.py              # 法线贴图烘焙
│   └── losses.py                    # Loss函数库
│
├── bridge/                          # Messiah通信桥接
│   ├── messiah_bridge.py            # 客户端 (优化器侧)
│   └── protocol.py                  # 通信协议定义
│
├── messiah_plugin/                  # Messiah Editor 插件
│   ├── __init__.py
│   ├── export_for_optimizer.py      # 场景导出
│   ├── import_optimized.py          # 导入优化结果
│   ├── optimizer_server.py          # JSON-RPC 服务端
│   └── install.py                   # 插件安装脚本
│
├── io/                              # 资源IO
│   ├── mesh_io.py                   # Mesh 导入导出 (glTF/FBX)
│   ├── texture_io.py               # 贴图 导入导出 (DDS/PNG/TGA)
│   └── material_io.py              # 材质 导入导出 (JSON/XML)
│
└── tests/
    ├── test_pipeline.py             # 管线测试
    ├── test_brdf.py                 # BRDF 一致性测试
    └── test_optimizer.py            # 优化器测试
```

---

## 九、实施路线图

### Phase 1: 基础可微管线 (2-3周)

- [x] 分析 Messiah BRDF/Shading 参数
- [ ] 实现 `MessiahDiffPipeline` (GGX BRDF + DefaultLit)
- [ ] 实现基础 Loss 函数 (L2 + Smoothness)
- [ ] 验证：对比 nvdiffrast 渲染 vs Messiah 渲染结果
- [ ] BRDF 一致性单元测试

### Phase 2: 贴图优化器 (2-3周)

- [ ] 实现 `TextureOptimizer`
- [ ] 贴图分辨率降级优化
- [ ] 法线贴图烘焙优化
- [ ] 多视角优化支持
- [ ] DDS/PNG/TGA 导入导出

### Phase 3: 外部编辑器 (3-4周)

- [ ] PyQt6 主窗口框架
- [ ] 3D 预览视口 (OpenGL Widget)
- [ ] 优化参数面板
- [ ] Loss 曲线实时监控
- [ ] Before/After 对比视图
- [ ] glTF/FBX 场景导入

### Phase 4: Shader 简化器 (2-3周)

- [ ] 实现更多 Messiah ShadingModel (SSS, Hair, Cloth 等)
- [ ] ShaderSimplifier 复杂→简单Shader拟合
- [ ] 材质参数自动搜索
- [ ] Perceptual Loss 集成

### Phase 5: Messiah 集成 (2-3周)

- [ ] Messiah Editor Python 插件: 场景导出/导入
- [ ] JSON-RPC 通信桥 (实时同步)
- [ ] 利用 ShaderWatcher 热重载优化结果
- [ ] 一键优化工作流
- [ ] 性能分析 & 内存优化

### Phase 6: 高级功能 (持续)

- [ ] 批量优化 (多物体/多材质)
- [ ] 更多 ShadingModel 支持 (Eye, Foliage, Water 等)
- [ ] IBL / 环境光照优化
- [ ] Cluster Shading 模拟
- [ ] 自动最优参数搜索 (bayesian optimization)
- [ ] 优化结果质量报告生成

---

## 十、关键技术挑战与解决方案

| 挑战 | 解决方案 |
|------|---------|
| **BRDF 一致性** - nvdiffrast 渲染必须匹配 Messiah | 逐项对照 BRDF.fxh / ShadingModel.fxh 实现，编写对比测试 |
| **GBuffer 差异** - Messiah 用延迟渲染，nvdiffrast 是前向 | 在可微管线中模拟 GBuffer 计算流程，但合并为单 pass |
| **Shader 复杂度** - 部分 Messiah Shader 包含分支和复杂计算 | 渐进实现：先 DefaultLit，再逐步添加 SSS/Hair/Cloth |
| **内存限制** - 高分辨率贴图 + batch 渲染消耗大量 GPU 内存 | 使用 gradient checkpointing, 分块渲染, mixed precision |
| **法线贴图** - TBN 空间变换的可微分实现 | 构建可微 TBN 矩阵，使用 nvdiffrast interpolate 的导数输出 |
| **色调映射** - Messiah 后处理链影响最终像素 | 在 Loss 计算前应用相同的 tonemapping (ACES/Reinhard) |
| **通信延迟** - 实时同步可能有延迟 | 使用共享内存或 mmap 传递大buffer，JSON-RPC 仅传控制信号 |

---

## 十一、依赖项

```
# requirements.txt
torch>=2.0.0
nvdiffrast>=0.3.1
PyQt6>=6.5.0
numpy>=1.24.0
Pillow>=10.0.0
torchvision>=0.15.0       # Perceptual Loss
pygltflib>=1.16.0         # glTF 导入导出
imageio>=2.31.0           # 图片 IO
matplotlib>=3.7.0         # Loss 曲线绘制
PyOpenGL>=3.1.6           # 3D 预览
pyyaml>=6.0               # 配置文件
```
