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
