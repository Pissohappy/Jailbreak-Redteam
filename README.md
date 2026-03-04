# VLM Redteam Skeleton

一个可运行的 LangGraph 多轮 beam-search VLM redteam demo。

## Requirements

- Python 3.11+

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 一键跑通 Demo

```bash
python -m vlm_redteam.cli.run --config configs/run.yaml --goal "Summarize this goal"
```

默认配置会：
- 使用内置 `demo_figstep` 攻击；
- 跑 `max_rounds=3`，`beam_width=2`，`per_branch_candidates=2`；
- 当 `target_vllm_base_url` 或 `judge_base_url` 为空时自动走 dummy 模式（但完整节点仍执行）。

运行后会产出：
- `runs/<run_id>/events.jsonl`
- `runs/<run_id>/artifacts/`（保存去重后的图片）
- `runs/<run_id>/generated_images/`（demo attack 生成的原始图片）

## 内置示例攻击

- 文件：`attacks/demo_figstep.py`
- 功能：将 goal 文本渲染到图片，并返回固定的安全 user prompt：
  - `Please describe the text shown in the provided image.`

该攻击仅用于框架联调演示，不包含任何越狱技巧。

## 接入用户现有攻击类（entrypoint + PYTHONPATH）

CLI 会从 `enabled_attacks` 读取攻击列表：
- `demo_figstep`：内置攻击
- `<module>:<ClassName>`：通过 entrypoint 动态导入用户攻击类

示例配置：

```yaml
enabled_attacks:
  - demo_figstep
  - my_attacks.figstep:FigStepAttack

attack_weights:
  demo_figstep: 1.0
  my_attacks.figstep:FigStepAttack: 2.0

attack_init_kwargs:
  my_attacks.figstep:FigStepAttack:
    output_image_dir: runs/demo-run-001/custom_images
```

运行时提供 `PYTHONPATH`：

```bash
PYTHONPATH=/path/to/your/code python -m vlm_redteam.cli.run --config configs/run.yaml
```

你的攻击类需提供 `generate_test_case(...)` 接口（可为现有 `BaseAttack/TestCase` 风格），框架会通过 `AttackAdapter` 统一成 dict 流程。

## 将 `attacks_strategy/` 与 `configs/attacks/` 接入当前仓库（推荐做法）

你现在的代码结构已经很接近可直接接入：

- 攻击实现：`attacks_strategy/<attack_name>/attack.py`
- 攻击参数：`configs/attacks/<attack_name>.yaml`
- expand 阶段的多方法选择：由 `enabled_attacks + attack_weights` 决定（加权采样）

### 1) 在 `run.yaml` 里启用多个攻击（决定 expand 可选方法集合）

```yaml
enabled_attacks:
  - attacks_strategy.figstep.attack:FigStepAttack
  - attacks_strategy.sd35_figstep.attack:SD35FigStepAttack
  - attacks_strategy.qr.attack:QRAttack

attack_weights:
  attacks_strategy.figstep.attack:FigStepAttack: 2.0
  attacks_strategy.sd35_figstep.attack:SD35FigStepAttack: 1.0
  attacks_strategy.qr.attack:QRAttack: 1.0
```

说明：
- `enabled_attacks` 里每个条目是 `<module>:<ClassName>`；
- `attack_weights` 控制 expand 里不同攻击被抽到的概率，值越大越容易被采样。

### 2) 把 `configs/attacks/*.yaml` 参数透传给具体攻击类初始化

你可以直接把每个攻击配置中的 `parameters` 填到 `attack_init_kwargs` 对应条目下（key 必须与 `enabled_attacks` 条目完全一致）：

```yaml
attack_init_kwargs:
  attacks_strategy.figstep.attack:FigStepAttack:
    output_image_dir: runs/demo-run-001/generated_images
    cfg:
      font_path: attacks/Arial Font/ARIAL.TTF
      font_size: 70
      wrap_width: 18
      steps: 3

  attacks_strategy.sd35_figstep.attack:SD35FigStepAttack:
    output_image_dir: runs/demo-run-001/generated_images
    cfg:
      stable_diffusion_path: /mnt/disk1/weights/t2i/stable-diffusion-3.5-medium
      guidance_scale: 7.0
      steps: 3
```

实践建议：
- 建一个“配置转换脚本”（如 `scripts/merge_attack_cfgs.py`），把 `configs/attacks/*.yaml` 的 `parameters` 自动转成 `attack_init_kwargs`，避免手工复制。
- 这样你就能继续维护独立攻击配置文件，同时又能无缝接入当前 CLI。

### 3) expand 如何“按预设策略”选攻击

当前 expand 节点会在每个分支里，按 `attack_weights` 对已注册攻击做带权采样（可重复），采样数由 `per_branch_candidates` 决定。

因此如果你想“按预设策略”运行，通常有两种方式：
- **静态策略**：直接调权重（例如 figstep:3, qr:1）；
- **分阶段策略**：每一轮/每一次 run 使用不同的 `run.yaml`（例如 round1 先高权重文字攻击，round2 再提高视觉扰动攻击权重）。

### 4) 运行方式

```bash
PYTHONPATH=. python -m vlm_redteam.cli.run --config configs/run.yaml --goal "your goal"
```

如果某些攻击依赖额外环境（如 `torch`, `numpy`, `core` 包等），请先在当前环境补齐依赖，否则动态导入会失败。
