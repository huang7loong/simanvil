# SimAnvil

<p align="left">
  <strong>面向具身智能的数字物理铁砧 —— 凹面资产锻造、三维受力检视、闭环物理测评与 VLA 数据清洗。</strong>
  <br/>
  <a href="README.md">English</a> · <a href="README_zh.md">简体中文</a> · <a href="docs/assets/simanvil_showcase.mp4">凹槽配盘对比视频 (.mp4)</a> · <a href="docs/assets/toyblocks_flip_forces.mp4">500Hz 翻滚受力视频 (.mp4)</a>
</p>

> [!NOTE]
> **项目定位与欢迎交流**：`SimAnvil` 源于我个人的具身智能实验工作台，初衷是把平时做 MuJoCo **仿真资产整理、轨迹排障测评与 VLA 数据合成**时反复踩坑的环节，沉淀成一套**轻量、实在、开箱即用**的工具箱。项目目前处于早期演进阶段（`v0.1`），如果你对这套工具流感兴趣，非常欢迎提 Issue 交流想法、提功能建议或一起完善！

![SimAnvil 同轨迹物理对照演示](docs/assets/hero_demo.gif)

### 500 Hz 积木顶边翻滚：三维接触力实时拆解（`0.2×` 慢放）

![积木翻滚三维接触力动态演示](docs/assets/toyblocks_flip_forces.gif)

---

## 为什么需要 SimAnvil？

在用 MuJoCo 为 VLA（视觉-语言-动作模型）合成操作数据时，单看普通 RGB 录像很容易漏掉 **3 类常见隐患**：

1. **“保鲜膜陷阱”（Shrink-Wrap Trap）**：把带凹槽的托盘、碗、U 型件直接导入 MuJoCo，默认会被单一外凸包封死洞口，形成看不见的“空气墙”；同时 Blender 默认导出的 Y-up `.glb` 与物理引擎 Z-up `.obj` 经常出现轴向错位。
2. **隐蔽的仿真作弊（Silent Reward Hacking）**：轨迹在相机画面里看似成功，实则依赖吸附焊接、异常高摩擦或状态瞬移，夹爪与工件接触力早已飙到上百牛顿却无从察觉。
3. **异常片段污染训练（Poisoned Trajectories）**：打滑重试、IK 奇异点卡顿、撞沿刮擦等尾部帧若按 `1.0×` 权重直接喂入 `LeRobot`，容易带偏策略；但若整条丢弃，又会浪费干净的前半段轨迹并破坏 Action Chunk 的时序连续性。

**`simanvil`** 采用 **“零大模型依赖的 Python 物理核心（`src/simanvil/`）+ AI Agent 技能层（`skills/`）”** 双模式设计：既能在离线算力集群直接跑批处理，也能接入 Antigravity、Claude Code、Cursor 或 Codex，辅助完成 **“资产建模 → 三维检视 → 归因调参 → 数据清洗导出”** 闭环。

---

## 零后端交互式三维物理检视台 (`simanvil inspect`)

任意录制轨迹均可一键导出为免安装的静态网页工作台，内置原生 Z-up `[0,0,1]` 坐标系、`[w,x,y,z] → [x,y,z,w]` 四元数同步、实时 `mj_contactForce` 三维力矢量箭头、`1/2/3` 外观与彩色凸包透视切换，以及基于 `GOP=1` 全关键帧编码的双机位逐帧同步拖拽：

| 凹槽配盘检视（28 块彩色 CoACD 凸包 + 接触力箭头） | 指尖推翻积木：接触力与表面字形检视 |
| :---: | :---: |
| ![CoACD 凸包透视](docs/assets/viewer_coacd_hulls.png) | ![ToyBlocks 接触力学检视](docs/assets/viewer_toyblocks_glyphs.png) |

---

## 架构设计：CLI 面向工程流水线，`skills/` 面向 AI Agent

```text
  原始网格 (.glb/.obj)             MuJoCo 仿真循环                    VLA 训练 (LeRobot / OpenPI)
          │                                │                                        ▲
          ▼                                ▼                                        │
  ┌───────────────┐              ┌───────────────────┐                    ┌───────────────────┐
  │ simanvil forge│ ──MjSpec───► │  AnvilRecorder    │ ──.npz + Δqpos=0─► │  simanvil assay   │
  │ Blender Z-up  │              │  (state, controls,│        │           │  0 斜率夹持 +      │
  │ + CoACD 凸分解 │              │   phase @ 500Hz)  │        │           │  0.15x 异常帧降权  │
  │ + 物理冒烟测试 │              └───────────────────┘        ▼           │  -> LeRobot v3    │
  └───────────────┘                                  ┌──────────────────┐ └───────────────────┘
                                                     │ simanvil inspect │
                                                     │ 三维力矢量箭头 +  │
                                                     │ 彩色凸包透视网页  │
                                                     └──────────────────┘
```

- **面向工程师与离线集群**：`src/simanvil/` 不依赖任何大模型 SDK，几何编译、物理回放校验与数据集构建均可纯本地确定性运行。
- **面向 AI 编程助手（Antigravity / Claude Code / Cursor / Codex）**：挂载 [`plugin.json`](plugin.json) 或 [`skills/`](skills/) 后，多模态 Agent 可直接调用 `simanvil` 完成凹面零件凸分解、排查仿真作弊与迭代控制器参数。

---

## 安装

```bash
# 基础核心包（Forge + Trace + Inspect —— 轻量安装，无需 PyTorch）
pip install -e .

# 可选扩展：开启 LeRobot v3 带权重数据集导出与 PyAV 解码读回校验
pip install -e ".[lerobot,dev]"
```

*注：`simanvil forge` 会调用本机无头 Blender 4.x（自动探测 macOS `/Applications/Blender.app`、`PATH` 中的 `blender` 或环境变量 `BLENDER_BIN`）。*

---

## 60 秒快速上手

### 1. 运行开箱即用的 CoACD 凸分解 vs. 单凸包对照演示

无需下载任何外部重型机器人资产：

```bash
python examples/quickstart_kitting.py
simanvil serve --dir outputs/quickstart/session --port 8895
# 浏览器打开 http://127.0.0.1:8895/index.html（快捷键：空格播放/暂停，1/2/3 切换凸包透视，F 开关力矢量）
```

### 2. 4 行代码接入现有 MuJoCo 仿真循环

`AnvilRecorder` 在仿真步进过程中不保存体积庞大的图像与接触力数组，只记录百 KB 级的 `(state, controls, phase)` 轨迹。离线导出阶段再确定性重建双机位视频与三维接触力，并自动执行 `Δqpos == 0` 动作重放一致性校验：

```python
from pathlib import Path
from simanvil import AnvilRecorder

rec = AnvilRecorder(model, data, task="拉开抽屉并平稳推回", robot_type="my_arm", fps=25)
for step in range(500):
    rec.step(ctrl, phase="S1_PULL/pull_handle")  # 阶段名含 retry / fault 将自动标记异常帧
rec.save(Path("outputs/runs/ep_001.npz"), passed=True)
```

随后一行命令即可导出三维网页检视台与带权重的 **`LeRobot v3`** 数据集：

```bash
simanvil inspect outputs/runs/ep_001.npz \
  --output outputs/session \
  --assay-out outputs/lerobot_v3
```

### 3. 在 PyTorch / VLA 训练中接入帧级权重 `sample_weight`

```python
per_frame_loss = criterion(pred_action, batch["action"])  # (B, ...)
weights = batch["sample_weight"].view(-1, *([1] * (per_frame_loss.ndim - 1)))
loss = (per_frame_loss * weights).sum() / weights.sum().clamp_min(1e-6)
```

---

## 核心物理与工程保证

| 关键环节 | `simanvil` 的机制保证 |
| :--- | :--- |
| **同源 Z-Up 微米级对齐** | 无头 Blender 统一 `ORIGIN_CENTER_OF_VOLUME` 体积重心原点，同步导出 `visual.glb` (`export_yup=False`) 与 `collision_source.obj` (`Y-forward, Z-up`)，强制校验双路包围盒误差 `< 1e-4 m`（典型值 `~0.2 µm`）。 |
| **MuJoCo 标准图层约定** | `group="2"` 视觉网格层（关闭碰撞）、`group="3"` CoACD 凸包碰撞层（默认 `condim="4"` 开启扭转摩擦防打滑）、`group="4"` 抓取与交互参考点 `<site>`。 |
| **位姿与四元数严格同步** | 前端显式固定 `THREE.Object3D.DefaultUp.set(0, 0, 1)`，严格按 MuJoCo `[w, x, y, z]` → Three.js `Quaternion(x, y, z, w)` 同步刚体位姿。 |
| **动作因果重放校验** | 从初始状态 `state[0]` 出发，仅输入录制的高频 `controls` 重新积分 `mj_step`，量化核对末态误差 `max_abs_qpos_err`。 |
| **逐帧无延迟拖拽与落盘核验** | 检视台相机视频采用 `PyAV` 全关键帧（`GOP=1`）H.264 编码；`LeRobot v3` 导出时自动抽帧解码校验 `MAE < 0.05`。 |
| **零侵入外部基准适配** | 通过 `register_episode_adapter` 即可接入第三方任务轨迹，无需修改核心代码（示例见 [`examples/inspect_toyblocks_dir.py`](examples/inspect_toyblocks_dir.py)）。 |

---

## 内置 Agent 技能 (`skills/`)

- [`skills/simanvil-forge/SKILL.md`](skills/simanvil-forge/SKILL.md) —— CoACD 凹度阈值调优、多部件铰接拆件规范与 `attach_forged_asset(MjSpec)` 挂载指南。
- [`skills/simanvil-inspect/SKILL.md`](skills/simanvil-inspect/SKILL.md) —— 仿真作弊（Reward Hacking）视觉与三维接触力排查清单。
- [`skills/simanvil-assay/SKILL.md`](skills/simanvil-assay/SKILL.md) —— 四级事后重标注（`G1–G3`）与异常帧处理策略（`downweight` / `zero-anomaly-frames` / `drop-failed-episodes`）选择指南。

---

## 演进路线图 (Roadmap)

- [x] **v0.1**：Blender 同源 Z-up + CoACD `forge`、轻量轨迹录制器 `AnvilRecorder`、多模型 HTML `inspect` 检视台、带权重 `LeRobot v3` `assay` 及配套 Agent `skills/`
- [ ] **v0.2**：多部件铰接 `.glb` 自动拆件与 `<joint type="hinge|slide"/>` 合成、可选 MuJoCo 4 Filament PBR (`mjrf`) 渲染后端、`.mcap` 真机日志导入适配器
- [ ] **v0.3**：无头浏览器多视角诊断图 CLI (`simanvil check-viewer`) 与 `mujoco_warp` GPU 批量域随机化接口
- [ ] **v0.4**：单图桌面场景重建工作流与真机两阶段安全空跑（Dry-Run）标定一致性校验

---

## 致谢与参考 (Acknowledgments)

`SimAnvil` 的设计深受具身智能社区优秀开源基准与工程实践的启发，在此特别致谢：

- **[AI2 ToyBlocks](https://github.com/allenai/ToyBlocks)** *(Allen Institute for AI)* —— 其声明式桌面积木操作基准、PDDL 符号几何谓词、非抓持顶边翻面任务及闭环状态校验机制，直接启发了本项目的轨迹数据结构设计与接触力学演示实验。
- **Google DeepMind Gemini Robotics 系列技术博客**（*Gemini 4 Argon for Robotics* 与 [*Gemini Robotics 2.5*](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)） —— 其关于接触对齐物理资产编译、免安装交互式 HTML 会话检视（防范 Reward Hacking）以及事后细粒度重标注与异常片段降权的数据飞轮理念，为本项目提供了重要的架构参考。
- 感谢底层物理仿真、几何处理与机器人学习开源生态：**[MuJoCo](https://github.com/google-deepmind/mujoco)**、**[CoACD](https://github.com/SarahWeiii/CoACD)**、**[Blender](https://www.blender.org/)** 与 **[LeRobot](https://github.com/huggingface/lerobot)**。

## 开源协议

Apache-2.0，详见 [LICENSE](LICENSE)。
