# NvDiffRast Messiah Optimizer — 使用指南

> 完整的分步操作指南，覆盖所有功能模块。

---

## 目录

1. [环境准备与安装](#1-环境准备与安装)
2. [启动应用](#2-启动应用)
3. [界面概览](#3-界面概览)
4. [独立模式：加载场景并优化](#4-独立模式加载场景并优化)
5. [引擎连接模式：与 Messiah Editor 联动](#5-引擎连接模式与-messiah-editor-联动)
6. [四种优化模式详解](#6-四种优化模式详解)
7. [参考图管理](#7-参考图管理)
8. [后处理匹配](#8-后处理匹配)
9. [迭代可视化面板](#9-迭代可视化面板)
10. [RenderDoc 管线对比](#10-renderdoc-管线对比)
11. [导出优化结果](#11-导出优化结果)
12. [配置文件说明](#12-配置文件说明)
13. [常见问题](#13-常见问题)

---

## 1. 环境准备与安装

### 1.1 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11（64-bit） |
| Python | 3.9 或更高（推荐 3.10） |
| GPU | NVIDIA GPU，支持 CUDA（推荐 RTX 2070 及以上） |
| CUDA Toolkit | 11.8 或 12.x |
| 可选 | RenderDoc（用于管线对比功能） |

### 1.2 安装步骤

**方式一：使用 pip 安装**

```bash
cd messiah_optimizer
pip install -r requirements.txt
```

依赖清单（`requirements.txt`）：
- `torch>=2.0.0` — PyTorch（含 CUDA 支持）
- `nvdiffrast>=0.3.1` — NVIDIA 可微分光栅化器
- `PyQt6>=6.5.0` — GUI 框架
- `numpy`, `Pillow`, `torchvision`, `pygltflib`, `imageio`, `matplotlib`, `PyOpenGL`, `pyyaml`

**方式二：使用 setup.py**

```bash
cd messiah_optimizer
pip install -e .
```

安装完成后，可通过命令行直接启动：

```bash
messiah-optimizer
```

### 1.3 安装 RenderDoc（可选）

如果需要使用 RenderDoc 管线对比功能：

1. 从 [https://renderdoc.org/](https://renderdoc.org/) 下载安装 RenderDoc
2. 确保 `renderdoccmd` 在系统 PATH 中（或记住其安装路径）
3. （可选）安装 RenderDoc Python 模块：`pip install renderdoc`（仅 replay 功能需要）

---

## 2. 启动应用

### 方式一：双击批处理脚本

双击 `run_optimizer.bat`，脚本会自动：
- 检查 Python 环境
- 检查并安装缺失依赖
- 启动优化器 GUI

### 方式二：命令行启动

```bash
cd messiah_optimizer
python editor/main.py
```

### 方式三：安装后启动

```bash
messiah-optimizer
```

启动后将看到深色主题的主窗口，布局包含 3D 视口、场景面板、优化面板等。

---

## 3. 界面概览

应用窗口由以下区域组成：

```
┌──────────────────────────────────────────────────────────────┐
│  菜单栏:  File | Engine | RenderDoc | View | Help            │
├──────────┬────────────────────────────────┬──────────────────┤
│          │                                │                  │
│  场景面板 │       3D 视口 / 图像对比        │   优化参数面板   │
│          │                                │                  │
│ (左侧)   │       (中央主区域)              │   (右侧)        │
│          │                                │                  │
├──────────┴────────────────────────────────┴──────────────────┤
│  参考图 │ 损失监控 │ 迭代历史 │ 导出 │ RenderDoc             │
│                     (底部停靠面板)                            │
├──────────────────────────────────────────────────────────────┤
│  状态栏: 连接状态、GPU 信息、当前迭代                          │
└──────────────────────────────────────────────────────────────┘
```

### 各面板功能速览

| 面板 | 功能 |
|------|------|
| **场景面板** (Scene Panel) | 浏览和加载场景文件；引擎模式下支持复选框选择性加载网格，右键贴图发起优化 |
| **3D 视口** (Viewport 3D) | 实时预览渲染结果，鼠标操控相机 |
| **图像对比** (Image Compare) | 并排/叠加/差异模式对比渲染与参考图 |
| **优化面板** (Optimization Panel) | 选择优化模式、设置参数、启停优化 |
| **参考图面板** (Reference Panel) | 管理优化目标参考图像 |
| **损失监控** (Monitor Panel) | 实时 Loss/PSNR 曲线可视化 |
| **迭代历史** (Iteration Panel) | 查看纹理演变、渲染对比、Shader 差异 |
| **导出面板** (Export Panel) | 导出优化后的纹理和材质 |
| **RenderDoc 面板** | 浏览 .rdc 捕获文件，检查渲染管线 |

---

## 4. 独立模式：加载场景并优化

这是最基本的使用方式，无需连接 Messiah 引擎。

### 步骤一：加载场景

1. **菜单 → File → Open Scene...** （快捷键 `Ctrl+O`）
2. 选择场景文件，支持以下格式：
   - **glTF / GLB**：标准 PBR 格式（推荐）
   - **OBJ**：通用网格格式（配合 .mtl 材质）
   - **FBX**：Autodesk FBX 二进制格式
   - **引擎资源**：Messiah `resource.xml` + `resource.data` 文件
3. 加载后，3D 视口会显示默认渲染结果
4. 场景层级树（Scene Panel 左侧）显示网格/材质结构

### 步骤二：操控 3D 视口

| 操作 | 相机行为 |
|------|---------|
| **左键拖拽** | 环绕旋转（Orbit） |
| **右键拖拽** | 平移（Pan） |
| **鼠标滚轮** | 缩放（Zoom） |

每次相机变化后，渲染会自动更新（30fps 节流）。

### 步骤三：加载参考图

1. 切换到 **参考图面板**（Reference Panel）
2. 点击 **"Load Reference"** 按钮
3. 选择目标外观的参考图像（PNG/JPG/EXR）
4. 图像将显示在面板中，并作为优化目标

> **提示**: 多视角优化时，可加载多张参考图（按 orbit 顺序排列）。

### 步骤四：配置优化参数

在 **优化面板** (Optimization Panel) 中：

1. **选择优化模式**：
   - `Texture` — 纹理优化（最常用）
   - `Material` — 材质参数拟合
   - `Shader` — Shader 简化
   - `Normal` — 法线贴图烘焙

2. **设置参数**（根据模式不同）：
   - **Learning Rate** — 学习率（默认 0.01）
   - **Iterations** — 迭代次数（默认 5000）
   - **Resolution** — 目标纹理分辨率
   - **Views** — 多视角数量（默认 16）

3. **配置损失权重**（Loss Weights 选项卡）：
   - `L2` — 像素级 L2 损失（默认 1.0）
   - `Perceptual` — VGG 感知损失（默认 0.1）
   - `SSIM` — 结构相似性（默认 0.05）
   - `Smoothness` — 纹理平滑正则化（默认 0.001）

4. **后处理匹配**（Comparison 选项卡）— 参见 [第 8 节](#8-后处理匹配)

### 步骤五：启动优化

1. 点击 **"Start Optimization"** 按钮
2. 优化过程中：
   - 3D 视口实时更新渲染结果
   - 损失监控面板显示 Loss/PSNR 曲线
   - 迭代历史面板记录每 N 步快照
   - 状态栏显示当前迭代和剩余时间
3. 点击 **"Stop"** 可随时停止
4. 优化完成后，视口显示最终结果

### 步骤六：导出结果

参见 [第 11 节](#11-导出优化结果)。

---

## 5. 引擎连接模式：与 Messiah Editor 联动

### 5.1 方式一：本地桥接（Local Bridge）— 推荐

适用于引擎代码可本地访问的场景。

#### 步骤一：设置引擎根目录

1. **菜单 → Engine → Set Engine Root...**
2. 选择引擎根目录，例如 `D:\NewTrunk\Engine\src\Engine`
3. 系统自动解析引擎资源结构（World/Repository）

#### 步骤二：选择世界并浏览网格列表

1. **菜单 → Engine → Pull Scene** 拉取世界
2. 在弹出的对话框中选择目标 `.iworld` 文件
3. 系统通过 `ResourceResolver` 解析：
   - `.iworld` / `.ilevel` XML 文件
   - GUID → 资源路径映射
   - Repository 索引（首次解析后缓存为 pickle）
4. 解析完成后，**场景面板左侧显示所有可用网格的复选框列表**
   - 每个网格名称旁有预览颜色标记（加载后用于视口高亮）
   - 状态栏显示可用网格总数

#### 步骤三：选择性加载网格

1. 在场景面板中 **勾选** 需要加载的网格（支持「全选」/「全不选」按钮）
2. 点击 **「▶ 加载选中网格」** 按钮
3. 系统仅加载选中的网格（无数量上限），跳过扁平网格过滤
4. 每个加载的网格在 3D 视口中以 **不同高亮颜色** 显示（30% 混合到 PBR 着色上）
5. 场景树自动展开，显示：
   - 各网格节点（带颜色标记和顶点数）
   - **每个网格下的关联贴图子节点**（通过依赖链自动发现）
   - 贴图角色自动检测：🎨 Base Color / 🔵 Normal Map / ⬛ Roughness 等

> **提示**: 可多次修改选择并重新加载，无需重新解析世界文件。

#### 步骤四：右键贴图发起优化

1. 在场景树中找到要优化的网格下的贴图节点
2. **右键点击** 该贴图，弹出上下文菜单：
   - **🎨 优化为 Base Map** — 以纹理优化模式优化此贴图
   - **🔵 优化为 Normal Map** — 以法线贴图烘焙模式优化
3. 选择后：
   - 优化面板自动切换到对应模式（Texture / Normal）
   - 原始贴图自动加载到参考图面板显示
   - 状态栏显示当前优化目标信息
4. 点击 **Start** 开始优化

#### 步骤五：捕获引擎参考帧

使用 RenderDoc 或 MPlatform.ScreenShot 获取引擎渲染截图作为优化参考。

### 5.2 方式二：TCP 桥接（Messiah Bridge）

适用于引擎编辑器运行中的实时交互。

#### 步骤一：安装引擎插件

1. **菜单 → Engine → Install Plugin**
2. 系统自动将 `messiah_plugin/` 下的脚本复制到引擎 Scripts 目录
3. 重启 Messiah Editor 生效（或使用热加载）

#### 步骤二：启动连接

1. 确保 Messiah Editor 已启动（插件会自动开启 RPC 服务，端口 9800）
2. **菜单 → Engine → Connect to Engine**
3. 状态栏显示连接状态

#### 步骤三：拉取场景

连接成功后可执行：
- **Pull Scene** — 从引擎拉取当前场景数据（Lua → JSON-RPC）
- **Capture Frame** — 通过 `MPlatform.ScreenShot` 截取引擎渲染帧
- **Sync Camera** — 同步优化器与引擎相机

#### 步骤四：推送结果

优化完成后：
- **Push Textures** — 将优化纹理写回引擎资源目录
- **Update Material** — 通过 Lua API 更新材质参数
- **Hot Reload** — 调用 `MResource.ReloadTexture()` 等 API 即时生效

---

## 6. 四种优化模式详解

### 6.1 纹理优化（Texture Optimization）

**目标**：优化 Base Color 纹理使渲染结果匹配参考图。

**工作原理**：
1. 在 logit 空间（`log(x/(1-x))`）参数化纹理，保证优化后像素值在 [0,1] 范围
2. 使用 Adam 优化器，cosine 学习率调度
3. 每次迭代随机选择一个视角渲染，计算复合损失，反向传播更新纹理

**操作步骤**：
1. 加载场景 → 加载参考图
2. 优化面板选择 **Texture** 模式
3. 设置目标分辨率（如 256×256 或 512×512）
4. 设置迭代次数（推荐 3000~5000）
5. 点击 **Start**

**参数建议**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Learning Rate | 0.01 | 过大导致振荡，过小收敛慢 |
| Iterations | 5000 | 简单场景 2000 即可 |
| Resolution | 256×256 | 越高细节越多，但优化越慢 |
| Views | 16 | 覆盖球面 360° |

### 6.2 材质拟合（Material Fitting）

**目标**：联合优化 Base Color、Roughness、Metallic 参数。

**工作原理**：
- 同时优化多个材质参数
- 不同参数使用差异化学习率（roughness/metallic 通常比 base_color 小）
- 支持 emission_strength 等附加参数

**操作步骤**：
1. 加载场景 → 加载参考图
2. 优化面板选择 **Material** 模式
3. 勾选要优化的参数（roughness / metallic / emission）
4. 设置迭代次数（推荐 2000）
5. 点击 **Start**

### 6.3 Shader 简化（Shader Simplification）

**目标**：将复杂着色模型（如 SSS、Cloth）拟合为 DefaultLit。

**工作原理**：
- 用 DefaultLit 渲染拟合 SSS/Cloth 着色结果
- 寻找最佳 DefaultLit 参数使两者视觉差异最小
- 用于降低运行时 Shader 复杂度

**操作步骤**：
1. 加载使用复杂 Shader 的场景
2. 优化面板选择 **Shader** 模式
3. 选择源 Shader 类型（SSS / Cloth）
4. 设置迭代次数（推荐 3000）
5. 点击 **Start**

### 6.4 法线贴图烘焙（Normal Map Baking）

**目标**：将高面数模型的表面细节烘焙到低面数模型的法线贴图上。

**工作原理**：
- 可微分渲染高面数模型获得多视角参考
- 优化低面数模型的法线贴图使渲染匹配
- 法线存储为切线空间

**操作步骤**：
1. 加载低面数目标网格
2. 使用高面数渲染结果或截图作为参考
3. 优化面板选择 **Normal** 模式
4. 设置法线贴图分辨率
5. 点击 **Start**

---

## 7. 参考图管理

### 加载参考图

1. 打开 **参考图面板** (Reference Panel)
2. 点击 **"Load Reference"**
3. 支持格式：PNG、JPG、BMP、TGA、EXR
4. 图片加载后显示缩略图

### 多视角参考

多视角优化需要为每个视角提供参考图：

1. 使用引擎截取多个角度的截图
2. 按 orbit 顺序（如 0°、22.5°、45°...）命名文件
3. 在参考图面板中按顺序加载
4. 或使用引擎连接自动捕获（Capture Frame 功能）

### 引擎截图作为参考

通过 TCP 桥接连接引擎后：

1. 在引擎中调整到目标视角
2. 在优化器中执行 **Capture Frame**
3. 通过 `MPlatform.ScreenShot` 获取引擎渲染结果
4. 自动设置为当前视角的参考图

---

## 8. 后处理匹配

### 为什么需要后处理匹配？

引擎渲染结果通常包含后处理效果（Bloom、色彩分级、暗角等），而优化器的可微分渲染管线默认不包含这些效果。直接对比会产生 **域差距**（Domain Gap），导致优化结果偏移。

### 后处理模式

在优化面板的 **Comparison** 选项卡中选择：

| 模式 | 说明 |
|------|------|
| `disabled` | 不使用后处理（默认，适用于无后处理的参考图） |
| `match_engine` | 使用预设引擎参数匹配 Messiah 后处理管线 |
| `custom` | 自定义后处理参数 |

### 可微分后处理效果

| 效果 | 参数 | 说明 |
|------|------|------|
| **DiffBloom** | threshold, intensity, radius | 高亮溢出效果 |
| **DiffColorGrading** | lift, gamma, gain, saturation | 色彩分级 |
| **DiffVignette** | intensity, roundness, smoothness | 画面暗角 |

### 设置步骤

1. 在优化面板 → Comparison 选项卡
2. 选择 Post-Process 模式
3. 如选 `custom`，调整各效果参数
4. 预览窗口实时显示带后处理的渲染结果
5. 启动优化 — 后处理效果参与梯度计算

---

## 9. 迭代可视化面板

迭代历史面板 (Iteration Panel) 提供三个选项卡，帮助你直观跟踪优化过程。

### 9.1 纹理演变 (Texture Evolution)

- 每隔 N 步保存纹理快照（自动间隔）
- 以缩略图画廊展示纹理从初始到最终的变化过程
- 点击任意快照查看大图

### 9.2 渲染对比 (Render Compare)

提供四种对比模式：

| 模式 | 说明 |
|------|------|
| **Side by Side** | 左右并排对比引擎参考与优化器渲染 |
| **Overlay Blend** | 叠加混合（可调滑块控制透明度） |
| **Difference** | 像素差异放大图 |
| **Reference Only** | 仅显示参考图 |

### 9.3 Shader 差异 (Shader Diff)

- GitHub 风格的参数变化对比
- HTML 渲染的差异视图
- 红色标记删除/旧值，绿色标记新增/新值
- 记录每步参数变化历史

---

## 10. RenderDoc 管线对比

RenderDoc 面板允许你直接检查引擎 GPU 渲染管线，是最精确的对比方式。

### 10.1 捕获 RenderDoc 帧

**方式一：手动捕获**
1. 使用 RenderDoc 打开 Messiah Editor
2. 按 F12（或配置的快捷键）捕获一帧
3. 保存 .rdc 文件

**方式二：通过优化器捕获**
1. 连接引擎后，**菜单 → RenderDoc → Capture Frame**
2. 系统调用 `renderdoccmd` CLI 进行捕获
3. .rdc 文件自动保存到工作目录

### 10.2 打开并浏览 .rdc 文件

1. 切换到 **RenderDoc 面板**
2. 点击 **"Open .rdc File"**
3. 选择捕获文件
4. 面板自动解析并显示：

#### Draw Calls 选项卡
- 完整的 Action Tree（Draw Call 层级）
- 每个 Draw Call 的事件 ID
- 点击任意 Draw Call → 右侧显示该时刻的 Framebuffer

#### Passes 选项卡
- 自动识别渲染 Pass（GBuffer、Lighting、PostProcess...）
- 基于 Action 名称和事件分组的智能识别
- 点击 Pass → 跳转到该 Pass 最后的 Draw Call

#### Textures 选项卡
- 列出帧中使用的所有纹理资源
- 显示纹理尺寸、格式信息
- 可选择纹理进行替换

### 10.3 提取 Framebuffer

1. 在 Draw Calls 中选择目标事件
2. 点击 **"Extract Framebuffer"**
3. 获取该时刻的渲染缓冲区图像
4. 可用作优化参考（比后处理后的截图更精确）

**典型用例**：提取 Lighting Pass 结束时的 Framebuffer（无后处理），作为纹理优化的参考。

### 10.4 纹理替换对比

1. 在 Textures 选项卡选择要替换的纹理
2. 点击 **"Load Replacement"** 加载替换纹理
3. 点击 **"Use Optimized"** 使用当前优化结果
4. 系统通过 RenderDoc Replay API 重放帧，使用替换纹理
5. 对比替换前后的渲染结果，验证优化效果

---

## 11. 导出优化结果

### 操作步骤

1. 切换到 **导出面板** (Export Panel)
2. 选择纹理格式：
   - **PNG** — 通用（推荐）
   - **DDS** — DirectX 纹理（引擎直接使用）
   - **TGA** — Targa 格式
   - **EXR** — HDR 格式
3. 选择材质格式：
   - **JSON** — 通用参数文件
   - **XML** — Messiah 引擎格式
4. 设置输出目录
5. 点击 **"Export"**

### 导出内容

根据优化模式不同，导出物包括：

| 优化模式 | 导出内容 |
|---------|---------|
| Texture | 优化后的 Base Color 纹理 |
| Material | 材质参数文件 + 优化纹理 |
| Shader | 简化后的 DefaultLit 参数 |
| Normal | 烘焙的法线贴图 |

### 热重载到引擎

如果已连接 Messiah Editor：

1. 导出完成后，点击 **"Hot Reload to Engine"**
2. 优化纹理自动复制到引擎资源目录
3. 通过 `MResource.ReloadTexture()` / `MEditor.Refresh()` 即时生效
4. 引擎视口立即反映变化，无需重启

---

## 12. 配置文件说明

### config/default_config.yaml

主配置文件，包含所有默认参数：

```yaml
rendering:
  resolution: [1024, 1024]    # 渲染分辨率
  device: "cuda"               # 计算设备（cuda / cpu）
  min_roughness: 0.08          # 最小粗糙度

optimization:
  texture:
    learning_rate: 0.01        # 学习率
    scheduler: "cosine"        # 调度器（cosine / step / exponential）
    max_iterations: 5000       # 最大迭代次数
    target_resolution: [256, 256]
    loss_weights:
      l2: 1.0
      perceptual: 0.1
      ssim: 0.05
      smoothness: 0.001
    num_views: 16              # 多视角数量

bridge:
  host: "127.0.0.1"
  port: 9800                   # TCP 桥接端口
  engine_root: ""              # 引擎根目录

export:
  texture_format: "png"
  material_format: "json"
  output_dir: "./output"
```

### config/user_config.yaml

用户覆盖配置，优先级高于 default_config。修改此文件不会影响默认值：

```yaml
bridge:
  engine_root: "D:\\NewTrunk\\Engine\\src\\Engine"
export:
  texture_format: "dds"
```

### config/messiah_brdf_params.yaml

Messiah BRDF 参数参考，定义着色模型的默认参数值。

---

## 13. 常见问题

### Q: 启动时报 "CUDA not available"

**A**: 检查 NVIDIA 驱动版本和 CUDA Toolkit 安装。运行 `python -c "import torch; print(torch.cuda.is_available())"` 验证。如无 GPU，系统会自动降级到 CPU 软件渲染器（速度慢但可用）。

### Q: 优化结果有色差

**A**: 通常是后处理域差距导致。尝试：
1. 启用后处理匹配（Comparison → `match_engine`）
2. 使用 RenderDoc 提取无后处理的 Framebuffer 作为参考
3. 调整 Perceptual Loss 权重

### Q: 连接引擎超时

**A**: 
1. 确认 Messiah Editor 已启动且插件已安装
2. 检查端口 9800 是否被占用（`netstat -an | findstr 9800`）
3. 检查防火墙设置

### Q: 纹理优化后模糊

**A**: 
1. 提高目标分辨率（如 512×512）
2. 增加迭代次数
3. 降低 Smoothness 损失权重

### Q: FBX 加载失败

**A**: 当前支持 FBX 二进制格式（v7100~7500+）。ASCII FBX 暂不支持，请在 DCC 工具中重新导出为二进制格式。

### Q: Repository 资源解析慢

**A**: 首次解析大型 Repository 需要扫描所有 XML 文件，会较慢。解析结果会缓存为 pickle 文件，后续启动秒级加载。

---

### Q: 选择性加载后视口中各网格颜色不同

**A**: 这是预期行为。选择性加载的每个网格会被分配一种独特的高亮色（蓝、橙、绿、粉等），以 30% 的权重混合到 PBR 着色结果上，帮助你区分不同网格。这不影响最终优化结果。

### Q: 右键贴图没有弹出菜单

**A**: 仅在「加载选中网格」后，场景树中的贴图子节点支持右键菜单。直接打开的 glTF/OBJ 场景暂不支持此功能。

---

> **文档版本**: v0.2.0  
> **最后更新**: 2026 年 3 月  
> **适用于**: NvDiffRast Messiah Optimizer v0.2.0
