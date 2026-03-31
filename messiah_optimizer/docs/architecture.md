# NvDiffRast Messiah Optimizer — 项目架构与原理文档

> 代码结构、模块设计、核心算法原理、数据流与关键实现细节。

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [代码目录结构](#3-代码目录结构)
4. [核心层：可微分渲染管线 (Pipeline)](#4-核心层可微分渲染管线-pipeline)
5. [优化层 (Optimizer)](#5-优化层-optimizer)
6. [IO 层 (io_utils)](#6-io-层-io_utils)
7. [桥接层 (Bridge)](#7-桥接层-bridge)
8. [编辑器层 (Editor)](#8-编辑器层-editor)
9. [引擎插件 (Messiah Plugin)](#9-引擎插件-messiah-plugin)
10. [配置系统](#10-配置系统)
11. [关键算法原理](#11-关键算法原理)
12. [数据流图](#12-数据流图)
13. [扩展指南](#13-扩展指南)

---

## 1. 项目概述

NvDiffRast Messiah Optimizer 是一个基于 NVIDIA nvdiffrast 的 **可微分渲染优化工具**，专为 Messiah 游戏引擎设计。核心思想是：

> 通过可微分光栅化建立一条从 **纹理/材质参数** 到 **最终像素** 的完整可导路径，利用梯度下降自动优化资产，使渲染输出匹配目标参考图像。

### 核心能力

| 能力 | 技术实现 |
|------|---------|
| 纹理优化 | Logit 空间参数化 + Adam + 多视角采样 |
| 材质拟合 | 差异化学习率联合优化多参数 |
| Shader 简化 | 复杂着色模型 → DefaultLit 映射 |
| 法线烘焙 | 高模 → 低模法线贴图可微传递 |
| 引擎集成 | TCP JSON-RPC + 本地资源解析 |
| 管线对比 | RenderDoc Replay API + 纹理替换 |

### 技术栈

```
PyTorch ← 自动微分引擎
  └─ nvdiffrast ← GPU 可微分光栅化（前向 + 反向）
     └─ CUDA ← 底层 GPU 计算
PyQt6 ← 桌面 GUI 框架
RenderDoc ← GPU 帧分析（可选）
```

---

## 2. 整体架构

系统采用 **分层架构**，自底向上依赖关系清晰：

```
┌─────────────────────────────────────────────────┐
│           Editor Layer (PyQt6 GUI)              │  ← 用户交互
│  main_window + 7 panels + 3 widgets             │
├─────────────────────────────────────────────────┤
│           Optimizer Layer                       │  ← 梯度优化
│  TextureOptimizer | MaterialFitter |            │
│  ShaderSimplifier | NormalMapBaker              │
├──────────────────┬──────────────────────────────┤
│  Pipeline Layer  │       Bridge Layer           │  ← 渲染 + 桥接
│  nvdiffrast +    │  Local/TCP Bridge +          │
│  PBR Shading     │  RenderDoc + ResourceResolver│
├──────────────────┴──────────────────────────────┤
│              I/O Layer                          │  ← 文件读写
│  Texture/Mesh/Material/FBX/Engine Loaders       │
├─────────────────────────────────────────────────┤
│           Messiah Plugin (Engine-side)           │  ← 引擎内运行
│  RPC Server + Export + Import + Hot Reload       │
└─────────────────────────────────────────────────┘
```

**设计原则**：
- **模块化**：每层独立可用，非 GUI 场景可直接调用 Pipeline + Optimizer
- **可微分端到端**：从纹理采样到最终像素的完整梯度链
- **格式适配**：翻译层处理 Messiah 特有格式，核心管线与引擎无关
- **渐进集成**：独立模式 → 本地桥接 → TCP 实时通信三级递进

---

## 3. 代码目录结构

```
messiah_optimizer/           # 项目根目录
├── editor/                  # GUI 编辑器（~13 文件，~3700 行）
│   ├── main.py              #   应用入口，创建 QApplication
│   ├── main_window.py       #   主窗口，协调所有面板和优化流程
│   ├── theme.py             #   深色主题样式表
│   ├── panels/              #   功能面板
│   │   ├── scene_panel.py   #     场景浏览与选择性加载(复选框+右键优化)
│   │   ├── optim_panel.py   #     优化参数配置
│   │   ├── reference_panel.py #   参考图管理
│   │   ├── monitor_panel.py #     Loss/PSNR 实时曲线
│   │   ├── export_panel.py  #     结果导出
│   │   ├── iteration_panel.py #   迭代历史可视化
│   │   └── renderdoc_panel.py #   RenderDoc 管线检查
│   └── widgets/             #   可复用控件
│       ├── viewport_3d.py   #     3D 渲染视口
│       ├── image_compare.py #     图像对比组件
│       └── loss_chart.py    #     损失曲线图表
│
├── pipeline/                # 可微分渲染管线（~9 文件，~2000 行）
│   ├── messiah_pipeline.py  #   核心7阶段渲染管线
│   ├── brdf.py              #   GGX PBR BRDF 实现
│   ├── shading_models.py    #   DefaultLit/SSS/Cloth 着色模型
│   ├── camera.py            #   相机系统与多视角生成
│   ├── tonemapping.py       #   色调映射（ACES/Reinhard/sRGB）
│   ├── postprocess.py       #   可微分后处理栈
│   ├── procedural.py        #   程序化网格/纹理生成
│   └── software_renderer.py #   CPU 软件渲染器（降级方案）
│
├── optimizer/               # 优化算法（~5 文件，~1200 行）
│   ├── losses.py            #   复合损失函数
│   ├── texture_optimizer.py #   纹理优化器
│   ├── material_fitter.py   #   材质参数拟合器
│   ├── shader_simplifier.py #   Shader 简化器
│   └── normal_baker.py      #   法线贴图烘焙器
│
├── io_utils/                # 文件 IO（~5 文件，~1800 行）
│   ├── texture_io.py        #   纹理读写（PNG/TGA/EXR/DDS）
│   ├── mesh_io.py           #   网格加载（glTF/OBJ）
│   ├── material_io.py       #   材质 IO（JSON/XML）
│   ├── fbx_loader.py        #   零依赖 FBX 二进制解析器
│   └── engine_scene_loader.py # Messiah resource.xml/data 加载 + 选择性加载 + 依赖链解析
│
├── bridge/                  # 引擎桥接（~9 文件，~2800 行）
│   ├── messiah_bridge.py    #   TCP JSON-RPC 客户端
│   ├── local_bridge.py      #   本地文件桥接服务
│   ├── protocol.py          #   通信协议（长度前缀帧）
│   ├── resource_resolver.py #   引擎资源 GUID 解析器
│   ├── renderdoc_capture.py #   RenderDoc 帧捕获
│   ├── rdoc_extractor.py    #   Framebuffer/纹理提取
│   ├── renderdoc_replay.py  #   RenderDoc Replay API 封装
│   └── unified_pipeline.py  #   统一资源+视觉对比协调器
│
├── messiah_plugin/          # 引擎侧插件（~5 文件，~700 行）
│   ├── optimizer_server.py  #   引擎内 JSON-RPC 服务器
│   ├── export_for_optimizer.py # 场景导出（Lua Bridge）
│   ├── import_optimized.py  #   结果导入 + 热重载
│   ├── install.py           #   插件安装/卸载脚本
│   └── optimizer_bridge_autostart.py # 自动启动钩子
│
├── config/                  # 配置文件
│   ├── default_config.yaml  #   默认配置
│   ├── messiah_brdf_params.yaml # BRDF 参数参考
│   └── user_config.yaml     #   用户覆盖配置
│
├── setup.py                 # 包安装配置
├── requirements.txt         # Python 依赖
└── run_optimizer.bat        # Windows 启动脚本
```

**总计**: ~52 个源文件，~11,000+ 行代码。

---

## 4. 核心层：可微分渲染管线 (Pipeline)

### 4.1 messiah_pipeline.py — 渲染核心

这是整个系统的心脏。`MessiahDiffPipeline` 类实现了一条完全可微分的 7 阶段 PBR 渲染管线：

```
输入(网格+纹理+相机+光照)
  │
  ▼
[Stage 1] Rasterize ──── nvdiffrast.rasterize()
  │                       输出: 三角形 ID + 重心坐标
  ▼
[Stage 2] Interpolate ── nvdiffrast.interpolate()
  │                       输出: 逐像素 UV、法线、世界坐标
  ▼
[Stage 3] Texture ────── nvdiffrast.texture()
  │                       输出: 逐像素 base_color, roughness, metallic, normal
  ▼
[Stage 4] Normal Map ─── 切线空间 → 世界空间法线变换
  │
  ▼
[Stage 5] Shade ──────── PBR 着色（BRDF 计算）
  │                       GGX + Schlick + Lambert/Burley
  ▼
[Stage 6] Antialias ──── nvdiffrast.antialias()
  │                       边缘抗锯齿（梯度也通过边缘传播）
  ▼
[Stage 7] Tonemap ────── ACES / sRGB gamma
  │
  ▼
[Stage 8] PostProcess ── DiffBloom + DiffColorGrading + DiffVignette
  │
  ▼
输出: [B, H, W, 3] 张量 (requires_grad=True)
```

**关键设计**：每个阶段都使用 PyTorch 运算或 nvdiffrast 的可微分操作，保证梯度可以从最终像素一路反传到输入纹理。

```python
# 简化的前向流程
def render(self, mesh, textures, camera, lights):
    # Stage 1-2: 光栅化 + 插值
    rast_out, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution)
    attr_out, _ = dr.interpolate(vtx_attr, rast_out, tri)

    # Stage 3: 纹理采样（可微分！）
    uv = attr_out[..., :2]
    base_color = dr.texture(textures['base_color'], uv)

    # Stage 5: PBR 着色
    color = self._shade_pbr(base_color, roughness, metallic, normal, ...)

    # Stage 6-7: 抗锯齿 + 色调映射
    color = dr.antialias(color, rast_out, pos_clip, tri)
    color = aces_tonemap(linear_to_srgb(color))

    return color  # 梯度流完整保留
```

### 4.2 brdf.py — BRDF 实现

精确复现 Messiah 引擎的 BRDF 计算（对应引擎 `BRDF.fxh`）：

| 函数 | 对应引擎 | 公式 |
|------|---------|------|
| `D_GGX(NoH, roughness)` | D_GGX in BRDF.fxh | $D = \frac{\alpha^2}{\pi((N \cdot H)^2(\alpha^2-1)+1)^2}$ |
| `F_Schlick(VoH, F0)` | F_Schlick | $F = F_0 + (1-F_0)(1-V \cdot H)^5$ |
| `Vis_Schlick_Disney(NoV, NoL, roughness)` | Vis_SmithJoint | 几何遮蔽项 |
| `Diff_Burley(NoV, NoL, VoH, roughness)` | Diffuse_Burley | Disney Burley 漫反射 |

这些函数全部使用 PyTorch 张量运算实现，天然支持自动微分。

### 4.3 shading_models.py — 着色模型

复现引擎 `ShadingModel.fxh` 中的四种着色模型：

```python
SHADING_MODELS = {
    0: 'Unlit',       # 无光照，直接输出 base_color
    1: 'DefaultLit',  # 标准 PBR（GGX Specular + Burley Diffuse）
    3: 'SSS',         # 次表面散射（添加散射项）
    10: 'Cloth',      # 布料（Charlie 分布 + 光泽层）
}
```

### 4.4 postprocess.py — 可微分后处理

解决引擎截图与优化器渲染之间的域差距：

```python
class PostProcessStack:
    """模式: disabled / match_engine / custom"""

    def forward(self, image):
        x = image
        if self.bloom.enabled:
            x = self.bloom(x)        # 高斯模糊提取高亮 → 叠加
        if self.color_grading.enabled:
            x = self.color_grading(x) # lift/gamma/gain 色彩映射
        if self.vignette.enabled:
            x = self.vignette(x)      # 径向衰减
        return x
```

所有后处理效果均使用可微分操作，梯度可穿透后处理栈传回纹理。

### 4.5 camera.py — 相机与多视角

```python
def create_orbit_cameras(num_views=16, distance=3.0, height=0.0):
    """在球面上均匀生成 num_views 个相机位置"""
    cameras = []
    for i in range(num_views):
        angle = 2 * pi * i / num_views
        eye = [distance*cos(angle), height, distance*sin(angle)]
        cameras.append(Camera(eye=eye, target=[0,0,0], up=[0,1,0], fov=45))
    return cameras
```

---

## 5. 优化层 (Optimizer)

### 5.1 losses.py — 复合损失函数

```python
class CompositeLoss:
    """加权组合多种损失"""
    components = {
        'l2':         (L2Loss,         1.0),    # 像素级
        'perceptual': (PerceptualLoss, 0.1),    # VGG16 特征空间
        'ssim':       (SSIMLoss,       0.05),   # 结构相似性
        'smoothness': (SmoothnessLoss, 0.001),  # 纹理平滑正则
    }
```

**PerceptualLoss** 使用预训练 VGG16 的 `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3` 特征层，提升结构与纹理匹配质量。

**SSIMLoss** = `1 - SSIM(rendered, reference)`，基于亮度、对比度、结构三个分量计算。

### 5.2 texture_optimizer.py — 纹理优化器

**核心技巧 — Logit 空间参数化**：

```python
# 问题：纹理像素值必须在 [0, 1] 范围内
# 方案：在 logit 空间优化，通过 sigmoid 映射回像素空间

logit_texture = torch.logit(initial_texture)  # [0,1] → (-∞, +∞)
logit_texture.requires_grad_(True)

optimizer = torch.optim.Adam([logit_texture], lr=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=max_iterations)

for i in range(max_iterations):
    texture = torch.sigmoid(logit_texture)  # (-∞, +∞) → [0, 1]
    rendered = pipeline.render(mesh, {'base_color': texture}, camera, lights)
    loss = composite_loss(rendered, reference)
    loss.backward()      # 梯度通过 sigmoid → pipeline → nvdiffrast 传回
    optimizer.step()
    scheduler.step()
```

### 5.3 material_fitter.py — 材质拟合器

联合优化多个材质参数，使用差异化学习率：

```python
param_groups = [
    {'params': [base_color],   'lr': 0.01},
    {'params': [roughness],    'lr': 0.005},  # 更保守
    {'params': [metallic],     'lr': 0.005},
]
optimizer = Adam(param_groups)
```

### 5.4 shader_simplifier.py — Shader 简化

将复杂着色模型映射到 DefaultLit：

```
SSS/Cloth 渲染结果 ──→ 参考图
DefaultLit(可优化参数) ──→ 渲染结果
Loss(DefaultLit渲染, 参考图) → backward → 更新 DefaultLit 参数
```

### 5.5 normal_baker.py — 法线贴图烘焙

```
高模渲染（多视角）──→ 参考图
低模 + 可优化法线贴图 ──→ 渲染结果
Loss → backward → 更新法线贴图
```

---

## 6. IO 层 (io_utils)

### 6.1 texture_io.py

- **加载**: PNG/JPG/TGA/EXR/DDS → PyTorch 张量 [H,W,C]
- **sRGB 处理**: 加载时 sRGB→linear（`pow(x, 2.2)`），保存时 linear→sRGB
- **自动 mipmap**: 通过 `torch.nn.functional.interpolate` 生成
- **格式统一**: 所有内部处理使用 float32 + linear 色彩空间

### 6.2 mesh_io.py

- **glTF/GLB**: 使用 `pygltflib` 解析，提取 positions/normals/uvs/indices
- **OBJ**: 自定义解析器，支持 v/vt/vn/f 语义
- **输出格式**: `{"vertices": [N,3], "triangles": [M,3], "normals": [N,3], "uvs": [N,2]}`
- **预处理**: 自动合并顶点属性为 `vtx_attr` 张量供 nvdiffrast 使用

### 6.3 fbx_loader.py — 零依赖 FBX 解析

完全自主实现的 FBX 二进制解析器（无需 Autodesk SDK）：

```
FBX 二进制格式：
  ├── Header (27 bytes): "Kaydara FBX Binary" + version
  ├── Node Records (recursive):
  │   ├── EndOffset + NumProperties + PropertyListLen + NameLen + Name
  │   ├── Properties (typed data: int/float/string/array...)
  │   └── Nested Nodes
  └── NULL Record (sentinel)

解析流程：
  1. 验证 magic bytes "Kaydara FBX Binary\x00"
  2. 读取版本号 (7100-7500+)
  3. 递归解析节点树
  4. 从 Geometry 节点提取:
     - Vertices (ByControlPoint double array)
     - PolygonVertexIndex (含负值编码多边形结束)
     - LayerElementNormal → 法线
     - LayerElementUV → UV 坐标
  5. 三角化多边形 (fan triangulation)
```

支持 DEFLATE 压缩数组的自动解压。

### 6.4 engine_scene_loader.py — Messiah 引擎资源加载

解析 Messiah 引擎特有的 `resource.xml` + `resource.data` 格式：

```
resource.xml:
  <Resource GUID="..." Type="StaticMesh">
    <LOD Index="0" VertexCount="..." IndexCount="..." .../>
    <Material GUID="..." Slot="0" .../>
  </Resource>

resource.data (二进制):
  [Vertex Buffer: float32 × VertexCount × Stride]
  [Index Buffer: uint16/uint32 × IndexCount]
  [Material Properties: ...]
```

**选择性加载** (`load_engine_scene_selective()`):

仅加载用户勾选的 GUID 对应网格（无数量上限），为每个网格分配高亮色并生成逐顶点 `vertex_colors` 张量，返回 `mesh_ranges` 用于视口高亮显示。

**依赖链解析** (`resolve_mesh_textures()`):

通过反向遍历 ResourceInfo 的 `deps` 字段，构建 Mesh → Material → Texture 的关联映射（Model.deps 包含 Mesh + Material，Material.deps 包含 Texture），并通过 `_guess_texture_role()` 从名称/路径推测贴图角色（base_color / normal / roughness 等）。

---

## 7. 桥接层 (Bridge)

### 7.1 resource_resolver.py — 资源 GUID 解析

Messiah 引擎使用 GUID 引用所有资源。ResourceResolver 建立 GUID → 文件路径的映射：

```python
class ResourceResolver:
    def __init__(self, engine_root):
        self.worlds_root = find("Worlds", engine_root)      # iworld/ilevel
        self.repo_root = find("Repository", engine_root)     # 资源仓库

    def build_index(self):
        """扫描 Repository 所有 .xml，建立 GUID → ResourceInfo 映射"""
        # 遍历 Repository/**/*.xml
        # 提取 GUID, Type, Path
        # 缓存为 pickle（首次慢，后续秒级）

    def resolve(self, guid) -> ResourceInfo:
        """GUID → { type, path, xml_path }"""
```

### 7.2 local_bridge.py — 本地桥接服务

不依赖引擎运行，直接读取引擎项目文件：

```python
class LocalBridgeServer:
    def start(self):
        self.resolver = ResourceResolver(engine_root)
        self.resolver.build_index()

    def list_worlds(self) -> List[WorldInfo]
    def load_world(self, world_path) -> SceneData
    def resolve_mesh(self, guid) -> MeshData
    def resolve_material(self, guid) -> MaterialData
```

### 7.3 messiah_bridge.py — TCP 客户端

与引擎内 `OptimizerRPCServer` 通信：

```python
class MessiahBridge:
    def connect(self, host="127.0.0.1", port=9800)
    def call(self, method, params) -> result   # JSON-RPC 2.0
    def export_scene(self) -> scene_json
    def capture_frame(self) -> image_path      # MPlatform.ScreenShot
    def import_texture(self, path, guid)
    def hot_reload(self)
```

### 7.4 protocol.py — 通信协议

长度前缀 JSON 帧（4 字节大端长度 + UTF-8 JSON 数据）：

```
[4 bytes: payload_length][payload_length bytes: JSON-RPC message]
```

### 7.5 RenderDoc 集成（三文件协作）

```
renderdoc_capture.py ── 负责捕获帧
  ├── In-process API (ctypes, RENDERDOC_API_1_0_0)
  └── CLI: renderdoccmd capture -w <pid>

rdoc_extractor.py ── 提取数据
  ├── renderdoccmd replay → 枚举纹理
  └── renderdoccmd replay --extract-framebuffer → PNG

renderdoc_replay.py ── 深度分析（最强大）
  ├── ReplayController: 打开 .rdc → 完整回放控制
  ├── Action Tree: 遍历所有 GPU 操作（Draw/Dispatch/Clear...）
  ├── Pass Identification: 基于 Action 名称自动识别 GBuffer/Lighting/PostProcess
  ├── Framebuffer Extraction: 在任意事件点提取渲染结果
  └── Texture Replacement: 替换纹理后重放，对比前后差异
```

### 7.6 unified_pipeline.py — 统一对比协调

```python
class UnifiedPipeline:
    def take_snapshot(self, name) -> Snapshot:
        """捕获当前状态（渲染结果 + 资源使用 + RenderDoc 帧）"""

    def compare_snapshots(self, before, after) -> ComparisonResult:
        """计算 PSNR/MSE/结构差异，生成差异图"""
```

---

## 8. 编辑器层 (Editor)

### 8.1 main_window.py — 中央协调器

`OptimizerMainWindow` 是整个 GUI 的核心，职责包括：

1. **布局管理**: Dock 面板排列与持久化
2. **配置加载**: `default_config.yaml` + `user_config.yaml` 合并
3. **优化循环**: QTimer 驱动的迭代循环
4. **渲染预览**: 相机变化 → 30fps 节流 → Pipeline.render()
5. **桥接管理**: LocalBridge / MessiahBridge 生命周期
6. **信号路由**: 面板间通信（Signal/Slot）

```python
class OptimizerMainWindow(QMainWindow):
    def __init__(self):
        # 初始化 Pipeline, Optimizer, Bridge
        # 创建所有面板并 addDockWidget
        # 连接信号

    def _optimization_step(self):
        """单次优化迭代（由 QTimer 触发）"""
        camera = random.choice(self.cameras)
        rendered = self.pipeline.render(mesh, textures, camera, lights)
        loss = self.loss_fn(rendered, reference)
        loss.backward()
        self.optimizer.step()
        # 更新 UI: 视口、监控、迭代历史
```

### 8.2 面板架构

所有面板继承 `QWidget`，通过 Qt Signal/Slot 与 MainWindow 通信：

```
ScenePanel ──(scene_loaded)──→ MainWindow ──→ Pipeline.load()
ScenePanel ──(mesh_selection_changed)──→ MainWindow ──→ load_engine_scene_selective()
ScenePanel ──(texture_optimize_requested)──→ MainWindow ──→ 切换优化模式 + 加载贴图预览
OptimPanel ──(start_clicked)──→ MainWindow ──→ _start_optimization()
MonitorPanel ←──(loss_updated)── MainWindow
IterationPanel ←──(snapshot_added)── MainWindow
RenderDocPanel ──(framebuffer_extracted)──→ ReferencePanel
```

### 8.3 theme.py — 深色主题

约 400 行 QSS（Qt Style Sheets），覆盖所有标准控件的深色样式。色调基于 `#1e1e1e` / `#2d2d30` / `#3e3e42` 灰度体系。

---

## 9. 引擎插件 (Messiah Plugin)

### 9.1 optimizer_server.py

在 Messiah Editor 内运行的 TCP 服务器，接收优化器请求：

```python
class OptimizerRPCServer:
    """JSON-RPC 2.0 服务器，监听端口 9800"""

    handlers = {
        'ping':             _handle_ping,
        'export_scene':     _handle_export_scene,      # MExecuter.sync(Lua)
        'capture_frame':    _handle_capture_frame,      # MPlatform.ScreenShot
        'import_texture':   _handle_import_texture,     # 复制文件
        'update_material':  _handle_update_material,    # Lua 更新
        'hot_reload':       _handle_hot_reload,         # MResource/MEditor 刷新
    }
```

### 9.2 引擎 API 调用链

```
优化器 → TCP → OptimizerRPCServer → MExecuter.sync(lua_code) → 引擎 Lua VM
                                   → MPlatform.ScreenShot()   → 引擎渲染器
                                   → MResource.ReloadTexture() → 资源管理器
                                   → MEditor.Refresh()         → 编辑器刷新
```

### 9.3 热重载流程

```
1. 优化器导出纹理到 output/
2. import_optimized.py:
   a. 备份原始纹理 (→ _backup/)
   b. 复制优化纹理到引擎资源目录
   c. 通过 Lua 更新材质参数
   d. 调用 MResource.ReloadTexture(guid) 刷新纹理缓存
   e. 调用 MEditor.Refresh() 刷新编辑器视口
```

---

## 10. 配置系统

采用两级 YAML 配置合并：

```python
# main_window.py
def _load_config(self):
    cfg = load("config/default_config.yaml")   # 完整默认值
    user = load("config/user_config.yaml")     # 用户覆盖
    deep_merge(cfg, user)                       # user 优先
    return cfg
```

运行时修改通过 `_save_user_config()` 持久化到 `user_config.yaml`，保持 `default_config.yaml` 不变。

---

## 11. 关键算法原理

### 11.1 可微分光栅化

nvdiffrast 的核心创新是使光栅化过程可微分：

**前向**: 标准 GPU 光栅化，输出三角形 ID + 重心坐标
**反向**: 通过 silhouette edges 处理不连续性，使梯度能穿过遮挡边界

```
常规光栅化: ∂pixel/∂vertex = 0（不可微）
nvdiffrast:  antialias() 在边缘引入 soft coverage → 梯度可以流过边缘
```

### 11.2 多视角采样优化策略

每次迭代随机选择一个视角，避免过拟合到单一角度：

$$L_{total} = \mathbb{E}_{i \sim \text{Uniform}(1..N)} \left[ L_{composite}(R(T, c_i), I_i^{ref}) \right]$$

其中 $R(T, c_i)$ 是使用纹理 $T$ 从相机 $c_i$ 的渲染结果，$I_i^{ref}$ 是对应参考图。

### 11.3 Logit 空间参数化

直接优化像素值容易越界。Logit 变换提供无约束优化空间：

$$T_{logit} = \log\frac{T}{1-T} \quad \text{(参数化)}$$
$$T = \sigma(T_{logit}) = \frac{1}{1+e^{-T_{logit}}} \quad \text{(还原)}$$

梯度通过 sigmoid 反传：$\frac{\partial T}{\partial T_{logit}} = T(1-T)$

### 11.4 PBR 着色微积分

Cook-Torrance 微表面 BRDF：

$$f_r = \frac{D \cdot F \cdot G}{4(N \cdot L)(N \cdot V)} + f_d$$

所有项使用 PyTorch 张量运算实现，保证全程可微。

### 11.5 感知损失

VGG16 特征空间的 L2 距离比像素空间更能反映人类视觉感知：

$$L_{perceptual} = \sum_{l \in \{relu1\_2, relu2\_2, relu3\_3, relu4\_3\}} \| \phi_l(I_{rendered}) - \phi_l(I_{reference}) \|_2$$

---

## 12. 数据流图

### 完整优化数据流

```
┌──────────────────────────────────────────────────────────┐
│                    用户操作 (Editor)                      │
│  加载场景 → 加载参考图 → 设置参数 → 点击 Start            │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│              I/O Layer: 解析文件                          │
│  glTF/OBJ/FBX/Engine → { vertices, triangles,            │
│                           normals, uvs, textures }       │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│           Pipeline: 可微分渲染                            │
│                                                          │
│  for iteration in range(max_iterations):                 │
│    cam = random_camera()                                 │
│    rendered = Pipeline.render(mesh, σ(logit_tex), cam)   │
│    loss = CompositeLoss(rendered, reference[cam])        │
│    loss.backward()  ←── 梯度反传 ──┐                     │
│    optimizer.step()                │                     │
│    scheduler.step()                │                     │
│                                    │                     │
│  梯度路径:                          │                     │
│  loss → tonemap → antialias →      │                     │
│  shade(BRDF) → texture_sample →    │                     │
│  interpolate → rasterize →         │                     │
│  sigmoid → logit_texture ◄─────────┘                     │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│           Export: 序列化结果                               │
│  σ(logit_tex) → PNG/DDS/TGA                              │
│  material_params → JSON/XML                              │
│  (可选) → 热重载到引擎                                    │
└──────────────────────────────────────────────────────────┘
```

### 引擎集成数据流

```
Messiah Editor                          Optimizer
     │                                      │
     │── [Plugin Install] ──────────────────→│
     │                                      │
     │←── [TCP Connect, port 9800] ─────────│
     │                                      │
     │── [export_scene → JSON] ─────────────→│ parse scene
     │                                      │
     │── [capture_frame → PNG] ─────────────→│ as reference
     │                                      │
     │                                      │ ... optimize ...
     │                                      │
     │←── [import_texture → file copy ] ────│
     │                                      │
     │←── [hot_reload → MResource API] ─────│
     │                                      │
     │   (viewport 立即更新)                 │
```

### RenderDoc 对比数据流

```
① 捕获：renderdoc_capture.py → frame.rdc
② 打开：renderdoc_replay.py → ReplayController
③ 遍历：Action Tree → 识别 Passes (GBuffer/Lighting/PostProcess)
④ 提取：选择事件 → GetFramebuffer() → NumPy array → 参考图
⑤ 替换：Load replacement texture → SetTextureOverride()
⑥ 重放：Replay with override → 新 Framebuffer
⑦ 对比：Before vs After → PSNR/差异图
```

---

## 13. 扩展指南

### 添加新的着色模型

1. 在 `pipeline/shading_models.py` 中添加新的着色函数
2. 在 `SHADING_MODELS` 字典中注册 ID → 名称映射
3. 在 `shade()` 分发函数中添加 case
4. 确保所有运算使用 PyTorch 张量操作（保持可微性）

### 添加新的优化目标

1. 在 `optimizer/` 下创建新的优化器类
2. 实现 `setup()`, `step()`, `get_result()` 接口
3. 在 `editor/panels/optim_panel.py` 中注册新模式
4. 在 `main_window.py` 中的优化调度逻辑添加分支

### 添加新的损失函数

1. 在 `optimizer/losses.py` 中添加新的 Loss 类
2. 实现 `forward(rendered, reference) -> scalar tensor`
3. 在 `CompositeLoss` 的 `components` 字典中注册
4. 在配置文件中添加默认权重

### 添加新的文件格式

1. 在 `io_utils/` 中添加加载器
2. 返回标准化格式 `{"vertices": ..., "triangles": ..., ...}`
3. 在 `mesh_io.py` 或 `texture_io.py` 的分发逻辑中注册扩展名

### 添加新的 Editor 面板

1. 在 `editor/panels/` 下创建 `xxx_panel.py`
2. 继承 `QWidget`
3. 在 `main_window.py` 中 import 并 `addDockWidget`
4. 通过信号连接到事件系统

---

> **文档版本**: v0.2.0  
> **最后更新**: 2026 年 3 月  
> **适用于**: NvDiffRast Messiah Optimizer v0.2.0
