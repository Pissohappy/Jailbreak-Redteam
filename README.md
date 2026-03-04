# VLM Redteam Skeleton

A minimal runnable repository skeleton for a future LangGraph-based multi-round beam-search VLM red-team framework.

## Requirements

- Python 3.11+

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run

```bash
python -m vlm_redteam.cli.run --config configs/run.yaml
```

Expected output:

- `Loaded config OK`
- `Graph compiled OK`

## 如何接入自定义攻击方法

你可以复用现有 `BaseAttack/TestCase` 风格攻击类（例如 `FigStepAttack`），通过 `AttackAdapter` 和 `AttackRegistry` 接入。

```python
from pathlib import Path
from random import Random

from vlm_redteam.attacks.adapter import AttackAdapter
from vlm_redteam.attacks.registry import AttackRegistry

# 1) 准备你的攻击对象（伪代码）
output_image_dir = Path("runs") / run_id / "generated_images"
attack_obj = FigStepAttack(cfg=cfg, output_image_dir=output_image_dir)

# 2) 包装为适配器并注册
adapter = AttackAdapter(attack_obj=attack_obj, attack_name="figstep")
registry = AttackRegistry()
registry.register("figstep", adapter, weight=1.0)

# 3) 生成 case_id（可按 run_id + round + 分支索引拼接）
case_id = f"{run_id}-r{round_idx}-b{branch_idx}-c{cand_idx}"

# 4) 传入 params 覆盖 cfg 同名字段（生成后会自动恢复）
params = {
    "alpha": 0.7,
    "steps": 20,
}

attack_name = registry.sample_attacks(k=1, rng=Random(42))[0]
test_case_dict = registry.get(attack_name).generate(
    original_prompt=original_prompt,
    image_path=original_image_path,
    case_id=case_id,
    params=params,
)
# 返回 dict 至少包含：
# case_id, jailbreak_prompt, jailbreak_image_path,
# original_prompt, original_image_path
```

说明：
- `output_image_dir` 由调用方在创建攻击对象时传入，便于和 `run_id` 对齐。
- `case_id` 建议在调度层统一生成，保证全局唯一且可追溯。
- `params` 会暂时覆盖 `attack_obj.cfg` 上同名字段，调用完成后恢复原值，避免跨候选污染。
