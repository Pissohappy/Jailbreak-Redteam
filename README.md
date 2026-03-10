# VLM Redteam Skeleton

一个可运行的 LangGraph 多轮 beam-search VLM redteam demo。

## 核心特性

- **真正的多轮对话**: 完整对话历史传递，目标模型能够理解之前的交互上下文
- **多模态历史支持**: 支持在多轮对话中传递图片历史
- **动态攻击策略**: 支持根据历史响应智能选择攻击方式（LLM-guided）
- **Beam Search**: 支持多分支并行探索，保留最优攻击路径

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
    config:
      font_path: attacks/Arial Font/ARIAL.TTF
      font_size: 70
      wrap_width: 18
      steps: 3

  attacks_strategy.sd35_figstep.attack:SD35FigStepAttack:
    output_image_dir: runs/demo-run-001/generated_images
    config:
      stable_diffusion_path: /mnt/disk1/weights/t2i/stable-diffusion-3.5-medium
      guidance_scale: 7.0
      steps: 3
```

实践建议：
- 建一个"配置转换脚本"（如 `scripts/merge_attack_cfgs.py`），把 `configs/attacks/*.yaml` 的 `parameters` 自动转成 `attack_init_kwargs`，避免手工复制。
- 这样你就能继续维护独立攻击配置文件，同时又能无缝接入当前 CLI。

### 3) expand 如何"按预设策略"选攻击

当前 expand 节点会在每个分支里，按 `attack_weights` 对已注册攻击做带权采样（可重复），采样数由 `per_branch_candidates` 决定。

因此如果你想"按预设策略"运行，通常有两种方式：
- **静态策略**：直接调权重（例如 figstep:3, qr:1）；
- **分阶段策略**：每一轮/每一次 run 使用不同的 `run.yaml`（例如 round1 先高权重文字攻击，round2 再提高视觉扰动攻击权重）。

### 4) 运行方式

```bash
PYTHONPATH=. python -m vlm_redteam.cli.run --config configs/run.yaml --goal "your goal"
```

如果某些攻击依赖额外环境（如 `torch`, `numpy`, `core` 包等），请先在当前环境补齐依赖，否则动态导入会失败。

## 多轮对话支持

框架已改造为**真正的多轮对话框架**，每次请求都会携带完整对话历史。

### 工作原理

1. **历史传递**: 每轮攻击时，目标模型会收到所有之前的 prompt 和 response
2. **多模态历史**: 历史消息中的图片会以 base64 格式嵌入请求
3. **Judge 评估**: Judge 仅评估当前轮次的攻击效果，不影响历史传递

### 数据流

```
Round 1:
  User: "攻击 prompt 1" + image1
  Target: "响应 1"

Round 2:
  History: [User: "攻击 prompt 1" + image1, Assistant: "响应 1"]
  User: "攻击 prompt 2" + image2
  Target: "响应 2"

Round 3:
  History: [User: "攻击 prompt 1" + image1, Assistant: "响应 1",
            User: "攻击 prompt 2" + image2, Assistant: "响应 2"]
  User: "攻击 prompt 3" + image3
  Target: "响应 3"
```

### Branch.history 结构

每个 Branch 维护一个 `history` 列表，记录所有历史交互：

```python
history = [
    {
        "round_idx": 1,
        "user_text": "攻击 prompt",
        "image_path": "/path/to/image.png",  # 可选
        "target_output": "目标模型的响应",
    },
    # ... 更多历史记录
]
```

### VLLMClient API

`VLLMClient` 支持多轮对话：

```python
from vlm_redteam.models.vllm_client import VLLMClient

client = VLLMClient(base_url="http://localhost:8000", model="model-name")

# 单轮调用（无历史）
response = client.generate_one(
    user_text="Hello",
    image_path=None,
    temperature=0.2,
    max_tokens=512,
)

# 多轮调用（带历史）
history = [
    {"user_text": "Hi", "image_path": None, "target_output": "Hello!"},
    {"user_text": "What is 2+2?", "image_path": None, "target_output": "4"},
]
response = client.generate_one(
    user_text="And 3+3?",
    image_path=None,
    temperature=0.2,
    max_tokens=512,
    history=history,  # 传入历史
)

# 批量调用也支持历史
outputs = client.generate_batch([
    {"user_text": "Q1", "image_path": None, "temperature": 0.2, "max_tokens": 512, "history": history1},
    {"user_text": "Q2", "image_path": None, "temperature": 0.2, "max_tokens": 512, "history": history2},
])
```

### 辅助函数

使用 `build_conversation_history()` 从 Branch.history 构建标准对话格式：

```python
from vlm_redteam.graph.state import build_conversation_history

messages = build_conversation_history(branch.history)
# 返回 OpenAI 格式的 messages 列表
```

## 架构说明

```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph Workflow                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │  Expand  │───▶│  Execute │───▶│  Judge   │───▶│ Select │ │
│  │ (生成攻击)│    │(调用目标) │    │ (评估)   │    │(更新beam)│ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│       │                                                │     │
│       │              ┌─────────────────────────────────┘     │
│       │              │                                       │
│       │              ▼                                       │
│       │        ┌──────────┐                                  │
│       └───────▶│  done?   │───▶ 结束或继续下一轮              │
│                └──────────┘                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

关键节点:
- Expand: 根据 attack_weights 采样攻击，生成候选 test case
- Execute: 调用目标模型（带完整历史），获取响应
- Judge: 评估当前轮次的攻击效果
- Select: 更新 beam，记录历史，决定是否终止
```

## 扩展开发

### 添加新攻击

1. 创建攻击类，实现 `generate_test_case()` 方法：

```python
class MyAttack:
    def generate_test_case(self, original_prompt: str, image_path: str | None, **kwargs) -> dict:
        return {
            "jailbreak_prompt": "modified prompt",
            "jailbreak_image_path": "/path/to/image.png",  # 可选
        }
```

2. 注册到配置文件：

```yaml
enabled_attacks:
  - my_module:MyAttack

attack_weights:
  my_module:MyAttack: 1.0
```

### 自定义 Expand 策略

实现 `ExpandStrategy` 接口：

```python
from vlm_redteam.graph.nodes.expand_strategies.base import ExpandStrategy, ExpandContext

class MyStrategy(ExpandStrategy):
    name = "my_strategy"

    def select_attacks(self, ctx: ExpandContext) -> list[tuple[str, dict]]:
        # ctx.branch.history 可用于分析历史响应
        # ctx.registry.list_attacks() 获取可用攻击
        # 返回 [(attack_name, params), ...]
        pass
```

配置使用：

```yaml
expand_strategy: my_strategy
llm_guide_config:
  base_url: http://localhost:8000
  model: guide-model
```
