# Messiah Bridge 架构设计文档

## 概览

NvDiffRast Shader/Texture Optimizer 通过 **Messiah Bridge** 与 Messiah 引擎项目交互。  
当前采用 **目标化资源解析 + RenderDoc 截帧对比** 作为主要方式。

| 模块 | 文件 | 作用 |
|------|------|------|
| **资源解析器** | `bridge/resource_resolver.py` | 解析 iworld → ilevel → repository XML，按 GUID 精准定位资源 |
| **RenderDoc 截帧** | `bridge/renderdoc_capture.py` | 程序化截帧、A/B 帧图对比 |
| **本地桥接** | `bridge/local_bridge.py` | 整合资源解析 + 截帧对比 + 文件读写 |
| TCP RPC 桥接（可选） | `bridge/messiah_bridge.py` | 需引擎 Python 面板可用（高级） |

---

## 1. 资源解析架构

### 1.1 为什么不直接扫描

Messiah 引擎的资源系统特点：
- 资源通过 **GUID** 引用，而非文件路径
- Repository 目录包含 **336万+** 条资源记录，全扫描不可行
- 每个世界 (World) 只引用其中一小部分资源

### 1.2 解析链路

```
.iworld (XML)
  │ <Level Name="$root"/>
  │ <Level Name="Heightmap_x0_y0"/>
  │ ...
  ▼
.ilevel (XML, ~1MB per file)
  │ <Entity>
  │   <Resource>05f6ad48-ee52-9a43-b474-e7d0ef0c4a3d</Resource>
  │   <DensityMap>...</DensityMap>
  │   <OverrideMaterial>...</OverrideMaterial>
  │ </Entity>
  │ 收集所有 GUID 引用
  ▼
resource.repository (XML, in Repository/*.local/)
  │ <Item>
  │   <GUID>05f6ad48-...</GUID>
  │   <Type>Texture</Type>
  │   <Name>env_ref_qinghe_16_base</Name>
  │   <Package>Env/Cubes/hex_world_W12K</Package>
  │   <SourcePath>..\..\LocalData\Cubes\..._2d.tga</SourcePath>
  │ </Item>
  ▼
ResourceInfo(guid, type, name, package, source_path, ...)
```

### 1.3 缓存策略

首次构建 Repository 索引约需 60-70 秒（解析 1336 个 .local 目录）。  
索引以 pickle 格式缓存到磁盘，后续加载仅 **~6 秒**。

缓存路径: `<Package>/.resolver_cache/repo_index_<hash>.pkl`

### 1.4 使用示例

```python
from bridge.resource_resolver import ResourceResolver

resolver = ResourceResolver(
    worlds_dir=r'I:\trunk_bjs\common\resource\Package\Worlds',
    repository_dir=r'I:\trunk_bjs\common\resource\Package\Repository',
)

# 列出可用世界（1242 个 .iworld 文件）
worlds = resolver.list_worlds()

# 选择一个世界，解析其所有资源 GUID
world = resolver.parse_world(worlds[0]['path'])
# → world.resource_guids: Set[str] (数百～数千个 GUID)

# 通过 Repository 索引解析 GUID → 实际资源信息
world = resolver.resolve_world(world)

# 按类型筛选
textures = resolver.get_textures(world)   # Texture 类型
meshes = resolver.get_meshes(world)       # Mesh/StaticMesh 类型
materials = resolver.get_materials(world) # Material 类型

# 查看统计
stats = resolver.get_world_stats(world)
# {'total_levels': 48, 'total_guids': 4160, 'resolved': 592,
#  'by_type': {'LodModel': 321, 'Texture': 103, ...}}
```

---

## 2. RenderDoc 截帧对比

### 2.1 用途

RenderDoc 用于**视觉 A/B 对比**：
1. 优化前截帧 → 提取 framebuffer 图片
2. 应用优化后截帧 → 提取 framebuffer 图片
3. 计算像素差异 (MSE, PSNR, 通道误差)

### 2.2 使用示例

```python
from bridge.renderdoc_capture import RenderDocCapture, CaptureWorkflow

rdoc = RenderDocCapture()  # 默认路径 C:\Program Files\RenderDoc
print(rdoc.is_available())  # True/False

# 提取 framebuffer
rdoc.extract_framebuffer('capture.rdc', 'output.png')

# A/B 对比
workflow = CaptureWorkflow(rdoc, './comparison')
workflow.capture_before('before.rdc')
workflow.capture_after('after.rdc')
result = workflow.compare()
# {'mse': 12.3, 'psnr': 37.2, 'max_pixel_diff': 45.0, ...}

# 在 RenderDoc GUI 中打开
rdoc.open_in_renderdoc('capture.rdc')
```

---

## 3. 本地桥接 (LocalBridgeServer)

整合 ResourceResolver + RenderDocCapture 的高层接口。

```python
from bridge.local_bridge import LocalBridgeServer

bridge = LocalBridgeServer(
    engine_root=r'D:\NewTrunk\Engine\src\Engine',
    worlds_dir=r'I:\trunk_bjs\common\resource\Package\Worlds',
    repository_dir=r'I:\trunk_bjs\common\resource\Package\Repository',
)
bridge.start()

# 选择世界（取代旧的全目录扫描）
stats = bridge.select_world(r'...\world_W4K_BJS_P3.iworld')

# 获取该世界的资源
textures = bridge.get_textures()
meshes = bridge.get_meshes()

# 推送优化结果到引擎（自动备份原文件）
bridge.import_texture('optimized.png', 'Content/Textures/base.png')
bridge.import_shader('optimized.hlsl', 'Shaders/PBR/skin.hlsl')

# RenderDoc 对比
bridge.capture_before('before.rdc')
bridge.capture_after('after.rdc')
diff = bridge.compare_captures()

bridge.stop()
```

### 3.1 API 参考

| 方法 | 说明 |
|------|------|
| `list_worlds()` | 列出所有 .iworld 文件 |
| `select_world(path)` | 选择世界并解析资源 GUID → 返回统计 |
| `get_textures/meshes/materials/effects()` | 获取当前世界的各类资源 |
| `export_scene(dir)` | 导出当前世界资源清单到目录 |
| `import_texture(src, target)` | 推送优化贴图（自动备份） |
| `import_shader(src, target)` | 推送优化着色器（自动备份） |
| `batch_import(dir, manifest)` | 批量推送优化结果 |
| `is_renderdoc_available()` | RenderDoc 是否可用 |
| `capture_before/after(rdc)` | 注册 A/B 截帧 |
| `compare_captures()` | 对比截帧差异 |
| `get_engine_info()` | 项目信息、世界数、RenderDoc 状态 |

---

## 4. 典型工作流

```
[启动优化器]
     │
     ├── 自动恢复 engine_root (user_config.yaml)
     │
[菜单: 从引擎读取场景]
     │
     ├── 弹出世界选择列表 (1242 个 .iworld)
     │   └── 用户选择 world_W4K_BJS_P3.iworld
     │
     ├── 解析 iworld → 48 个 Level
     ├── 解析 48 个 ilevel → 4160 个 GUID
     ├── Repository 索引查找 → 592 个资源解析成功
     │   (103 Texture, 321 LodModel, 46 ParticleSystem, ...)
     │
[预览/选择要优化的资源]
     │
     ├── 选择贴图 → nvdiffrast 可微渲染优化
     ├── 选择着色器 → 参数优化
     │
[RenderDoc A/B 对比]
     │
     ├── 优化前: 手动在引擎中截帧 → before.rdc
     ├── 优化后: 推送结果 → 引擎截帧 → after.rdc
     └── 自动对比: MSE / PSNR / 差异图
```

---

## 5. 文件结构

```
messiah_optimizer/bridge/
├── __init__.py              # 导出所有模块
├── resource_resolver.py     # iworld/ilevel/repository XML 解析
├── renderdoc_capture.py     # RenderDoc 截帧 + A/B 对比
├── local_bridge.py          # 整合桥接（推荐入口）
├── messiah_bridge.py        # TCP RPC 客户端（高级可选）
└── protocol.py              # JSON-RPC 线协议（高级可选）
```

## 6. 关键路径

| 路径 | 内容 |
|------|------|
| `I:\trunk_bjs\common\resource\Package\Worlds` | 1242 个 .iworld + 数千 .ilevel |
| `I:\trunk_bjs\common\resource\Package\Repository` | 1336 个 .local 目录，336万资源 |
| `D:\NewTrunk\Engine\src\Engine` | 引擎项目根目录 |
| `C:\Program Files\RenderDoc` | RenderDoc 安装目录 |
     ├── 弹出文件夹选择对话框
     ├── 用户选择引擎根目录（如 D:\NewTrunk\Engine\src\Engine）
     ├── 验证 Editor/ 子目录存在
     ├── 创建 LocalBridgeServer
     ├── 保存到 user_config.yaml
     └── 状态栏: "已连接引擎项目: ... (40 资源, 427 着色器)"
```

---

## 3. TCP RPC 桥接（高级可选）

### 3.1 适用场景

当需要以下实时操作时，可启用 TCP RPC 模式：
- 引擎端相机同步
- 触发引擎热更新（MEditor.RefreshResources / MRender.RefreshShaderSource）
- 在引擎中执行 Lua 脚本（MExecuter.sync）
- 捕获引擎渲染截图

### 3.2 前置条件

1. Messiah 引擎编辑器支持 Python 脚本面板（并非所有版本均有）
2. 已安装引擎端插件：`python messiah_plugin/install.py <引擎根目录>`
3. 在引擎编辑器中手动激活 Python 面板以启动 RPC 服务（端口 9800）

### 3.3 架构

```
┌─────────────────────────────┐      TCP (JSON-RPC 2.0)      ┌─────────────────────────────┐
│  NvDiffRast Optimizer       │      port 9800                │  Messiah Engine Editor      │
│  (独立 Python 进程)          │◄════════════════════════════►│  (C++ + 内嵌 Python)         │
│                             │                               │                             │
│  ┌───────────────────┐      │   [4字节长度头][JSON payload]  │  ┌───────────────────┐      │
│  │ MessiahBridge     │──────┼──────────────────────────────►│  │ OptimizerRPCServer│      │
│  │ (socket 客户端)    │◄─────┼──────────────────────────────│  │ (socket 服务端)    │      │
│  └───────────────────┘      │                               │  └────────┬──────────┘      │
│                             │                               │           │                 │
│                             │                               │  ┌────────▼──────────┐      │
│                             │                               │  │ MEditor / MEngine  │      │
│                             │                               │  │ MExecuter.sync()   │      │
│                             │                               │  └───────────────────┘      │
└─────────────────────────────┘                               └─────────────────────────────┘
```

### 3.4 通信协议

每条消息由 **4字节小端长度头** + **UTF-8 JSON body** 组成（JSON-RPC 2.0 规范）：

```
┌──────────────┬──────────────────────────────────────────────┐
│ 4 bytes (LE) │        N bytes UTF-8 JSON                    │
│ msg length   │ {"jsonrpc":"2.0","method":"...","params":{}} │
└──────────────┴──────────────────────────────────────────────┘
```

### 3.5 RPC 方法

| 方法 | 方向 | 用途 |
|------|------|------|
| `ping` | → 引擎 | 连接验证 |
| `export_scene` | ← 引擎 | 导出场景数据 |
| `capture_frame` | ← 引擎 | 渲染当前视角截图 |
| `capture_multiview` | ← 引擎 | 环绕渲染多视角参考图 |
| `import_texture` | → 引擎 | 推送优化贴图 |
| `update_material` | → 引擎 | 更新材质参数 |
| `hot_reload` | → 引擎 | 触发着色器/资源热更新 |
| `camera_update` / `get_camera` | ↔ 引擎 | 相机同步 |

### 3.6 已知限制

- **Python 面板可能不可用**：`qtmain.init()` 仅在用户点击编辑器 Python 🐍 按钮后执行，部分编辑器版本没有此按钮
- **site-packages 自启动无效**：引擎嵌入式 Python 跳过了 `site` 模块，`.pth` 文件不生效
- 因此本方案仅作为**高级可选**模式保留，主要工作流使用本地文件桥接

---

## 4. Messiah 引擎 Python API 参考

以下 API 已通过引擎 C++ 源码确认存在（`Sources/Runtime/Plugins/Python/Interface/Modules/`）。
仅在 TCP RPC 模式下使用。

### 4.1 已确认模块

| 模块 | 用途 | 函数数量 |
|------|------|---------|
| `MExecuter` | 跨线程 Python/Lua 执行 | 核心桥接 |
| `MEditor` | 编辑器功能 | ~17 |
| `MRender` | 渲染相关 | 100+ |
| `MResource` | 资源管理 | ~47 |
| `MEngine` | 运行时核心 | ~54 |
| `MObject` | 对象操作 | ~12 |
| `MConsole` | 控制台集成 | ~3 |

### 4.2 关键 API

**MExecuter（核心桥接）：**
- `MExecuter.sync(code: str, returnType: int, returnHint: str) -> str`
  - returnType: 0=void, 1=int, 2=string, 3=float
  - 唯一完全确认签名的 API，通过它执行 Lua 代码完成大部分操作

**MEditor：**
- `MEditor.RefreshResources(paths)` — 刷新资源引用 ✅
- `MEditor.GetCameraTransformFromAffiliatedResourceView(...)` — 相机变换

**MRender：**
- `MRender.RefreshShaderSource()` — 重新加载所有 Shader ✅
- `MRender.SaveAffiliatedViewportTexture(...)` — 保存视口截图
- `MRender.CaptureSequence(...)` — 序列捕获

**MResource：**
- `MResource.RefreshResourceByPath(path)` — 刷新单个资源 ✅

> **注意：** 除 `MExecuter.sync()` 外，其他函数的具体参数签名需在引擎控制台中通过 `help()` 验证。

---

## 5. 项目结构

```
messiah_optimizer/bridge/
├── __init__.py
├── local_bridge.py          # ← 主要：本地文件桥接
├── messiah_bridge.py        # ← 高级可选：TCP RPC 客户端
└── protocol.py              # ← 高级可选：JSON-RPC 线协议

messiah_optimizer/editor/main_window.py
└── Bridge 菜单:
    ├── 连接引擎项目...       → _on_bridge_connect()  → LocalBridgeServer
    ├── 断开连接              → _on_bridge_disconnect()
    ├── 从引擎读取场景         → _on_bridge_pull()     → export_scene()
    ├── 推送优化结果到引擎     → _on_bridge_push()     → batch_import()
    └── 引擎项目信息          → _on_bridge_info()     → get_engine_info()

messiah_plugin/                          # 仅 TCP RPC 模式需要
├── install.py                           # 引擎端插件安装
├── optimizer_server.py                  # 引擎端 RPC 服务
├── export_for_optimizer.py              # 场景导出
└── import_optimized.py                  # 结果导入
```
