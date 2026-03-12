#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# 动态实验运行脚本
#
# 功能：
#   1. 用vllm拉起target model服务（GPU1）
#   2. 用vllm拉起llm-guide model服务（GPU3）
#   3. Judge model 使用远程API (gpt-4o-mini)
#   4. 等待服务就绪后运行实验
#   5. 实验完成后清理服务
#
# 实验类型：
#   - exp5: 固定 llm-guided=gpt-4o-mini 和 judge=gpt-4o-mini，测试不同 target models
#   - exp6: 固定 target=Qwen3-VL-30B-A3B-Instruct 和 judge=gpt-4o-mini，测试不同 LLM-guided models
#
# Usage:
#   ./scripts/run_dynamic_experiments.sh --exp exp5
#   ./scripts/run_dynamic_experiments.sh --exp exp6
#   ./scripts/run_dynamic_experiments.sh --exp exp5 --dry-run
#   ./scripts/run_dynamic_experiments.sh --list-models
# ============================================================================

set -e

# ============================================================================
# 硬编码配置 - Python 环境
# ============================================================================

VLLM_PYTHON_PATH="/mnt/disk1/szchen/miniconda3/envs/vllm_env/bin/python"
KIMI_PYTHON_PATH="/mnt/disk1/szchen/miniconda3/envs/kimi_env/bin/python"

# ============================================================================
# 硬编码配置 - 模型列表
# ============================================================================
# 格式说明：
#   本地模型: "模型名|模型路径|vllm端口"
#   远程API:  "模型名|base_url|api_key"
# 使用 | 作为分隔符，避免 URL 中 :// 的解析问题
# ============================================================================

# Target Models (放在 GPU1)
TARGET_MODELS=(
    "Kimi-VL-A3B-Instruct|/mnt/disk1/weights/vlm/Kimi-VL-A3B-Instruct|8005"
    "GLM-4.1V-9B-Thinking|/mnt/disk1/weights/vlm/GLM-4.1V-9B-Thinking|8006"
    "gemma-3-27b-it|/mnt/disk1/weights/vlm/gemma-3-27b-it|8007"
    "Qwen3-VL-30B-A3B-Instruct|/mnt/disk1/weights/vlm/Qwen3-VL-30B-A3B-Instruct|8008"
    "deepseek-vl2|/mnt/disk1/weights/vlm/deepseek-vl2|8009"
)

# LLM Guide Models
GUIDE_MODELS=(
    "Kimi-VL-A3B-Instruct|/mnt/disk1/weights/vlm/Kimi-VL-A3B-Instruct|9005"
    "GLM-4.1V-9B-Thinking|/mnt/disk1/weights/vlm/GLM-4.1V-9B-Thinking|9006"
    "gemma-3-27b-it|/mnt/disk1/weights/vlm/gemma-3-27b-it|9007"
    "Qwen3-VL-30B-A3B-Instruct|/mnt/disk1/weights/vlm/Qwen3-VL-30B-A3B-Instruct|9008"
    "deepseek-vl2|/mnt/disk1/weights/vlm/deepseek-vl2|9009"
    "gpt-4o-mini|https://api.whatai.cc|sk-WE6FQhZoEr4nBz4KTqHuzAYixq6pQG07VZbRhcHaWdGtMTs2"
)

# Judge Model (使用远程API，不需要vllm)
JUDGE_BASE_URL="https://api.whatai.cc"
JUDGE_MODEL="gpt-4o-mini"
JUDGE_API_KEY="sk-WE6FQhZoEr4nBz4KTqHuzAYixq6pQG07VZbRhcHaWdGtMTs2"

# exp6 固定的 Target Model
EXP6_FIXED_TARGET="Qwen3-VL-30B-A3B-Instruct"

# GPU 配置
TARGET_GPU=1
GUIDE_GPU=3

# vLLM 配置
VLLM_DTYPE="bfloat16"
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_NUM_SEQS=256

# 实验配置
DATASET="/mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM/dataset/data_sample50.json"
BASE_CONFIG="configs/experiments/exp1_llm_guided.yaml"
EXPERIMENT_NAME="dynamic-experiment"
MAX_ROUNDS=3
BEAM_WIDTH=1
PER_BRANCH_CANDIDATES=1

# ============================================================================
# 全局变量
# ============================================================================
ACTIVE_PIDS=()
SKIP_CLEANUP=false
DRY_RUN=false

# ============================================================================
# 工具函数
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --exp TYPE               实验类型: exp5 或 exp6
                             exp5: 固定guide/judge=gpt-4o-mini，测试不同target models
                             exp6: 固定target=Qwen3-VL-30B-A3B-Instruct，测试不同guide models
    --dataset PATH           数据集路径 (default: $DATASET)
    --skip-cleanup           实验完成后不停止vllm服务
    --dry-run                只打印命令，不实际执行
    --list-models            列出所有可用模型
    -h, --help               显示帮助信息

Examples:
    # Exp5: 测试不同 target models
    $0 --exp exp5

    # Exp6: 测试不同 guide models
    $0 --exp exp6

    # 测试模式
    $0 --exp exp5 --dry-run

    # 查看模型列表
    $0 --list-models
EOF
}

list_models() {
    echo "=========================================="
    echo "Target Models (GPU$TARGET_GPU):"
    echo "=========================================="
    for i in "${!TARGET_MODELS[@]}"; do
        local spec="${TARGET_MODELS[$i]}"
        local name url_or_path port_or_key
        parse_model_spec "$spec" name url_or_path port_or_key
        if is_remote_api "$spec"; then
            echo "  [$i] $name (Remote API)"
            echo "      Base URL: $url_or_path"
            echo "      API Key: ${port_or_key:0:20}..."
        else
            local python_env="vllm_env"
            if [[ "$name" == *"Kimi"* ]]; then
                python_env="kimi_env"
            fi
            echo "  [$i] $name (Local)"
            echo "      Path: $url_or_path"
            echo "      Port: $port_or_key"
            echo "      Python Env: $python_env"
        fi
    done

    echo ""
    echo "=========================================="
    echo "LLM Guide Models:"
    echo "=========================================="
    for i in "${!GUIDE_MODELS[@]}"; do
        local spec="${GUIDE_MODELS[$i]}"
        local name url_or_path port_or_key
        parse_model_spec "$spec" name url_or_path port_or_key
        if is_remote_api "$spec"; then
            echo "  [$i] $name (Remote API)"
            echo "      Base URL: $url_or_path"
            echo "      API Key: ${port_or_key:0:20}..."
        else
            local python_env="vllm_env"
            if [[ "$name" == *"Kimi"* ]]; then
                python_env="kimi_env"
            fi
            echo "  [$i] $name (Local, GPU$GUIDE_GPU)"
            echo "      Path: $url_or_path"
            echo "      Port: $port_or_key"
            echo "      Python Env: $python_env"
        fi
    done

    echo ""
    echo "=========================================="
    echo "Judge Model (Remote API):"
    echo "=========================================="
    echo "  $JUDGE_MODEL"
    echo "      Base URL: $JUDGE_BASE_URL"
    echo "      API Key: ${JUDGE_API_KEY:0:20}..."

    echo ""
    echo "=========================================="
    echo "实验配置:"
    echo "=========================================="
    echo "  Exp5: 固定 guide/judge = gpt-4o-mini，遍历所有 target models"
    echo "  Exp6: 固定 target = $EXP6_FIXED_TARGET，遍历所有 guide models"

    echo ""
    echo "=========================================="
    echo "Python Environments:"
    echo "=========================================="
    echo "  vllm_env: $VLLM_PYTHON_PATH"
    echo "  kimi_env: $KIMI_PYTHON_PATH"
}

# 解析模型规格
# 格式: "name|url_or_path|port_or_key"
parse_model_spec() {
    local spec="$1"
    local -n _name="$2"
    local -n _url_or_path="$3"
    local -n _port_or_key="$4"

    IFS='|' read -r _name _url_or_path _port_or_key <<< "$spec"
}

# 判断是否为远程API
is_remote_api() {
    local spec="$1"
    local name url_or_path port_or_key
    IFS='|' read -r name url_or_path port_or_key <<< "$spec"
    # 如果 url_or_path 以 http:// 或 https:// 开头，则是远程API
    if [[ "$url_or_path" == http://* ]] || [[ "$url_or_path" == https://* ]]; then
        return 0
    else
        return 1
    fi
}

# 根据模型名称选择Python环境
get_python_path() {
    local model_name="$1"
    if [[ "$model_name" == *"Kimi"* ]]; then
        echo "$KIMI_PYTHON_PATH"
    else
        echo "$VLLM_PYTHON_PATH"
    fi
}

wait_for_server() {
    local url="$1"
    local max_wait=300
    local wait_interval=5
    local elapsed=0

    log "Waiting for server at $url to be ready..."

    while [ $elapsed -lt $max_wait ]; do
        if curl -s "$url/health" > /dev/null 2>&1; then
            log "Server at $url is ready!"
            return 0
        fi
        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))
        log "Still waiting... (${elapsed}s / ${max_wait}s)"
    done

    error "Server at $url failed to start within ${max_wait}s"
    return 1
}

start_vllm_server() {
    local name="$1"
    local model_path="$2"
    local port="$3"
    local gpu="$4"
    local extra_args="$5"

    # 根据模型名称选择Python环境
    local python_path=$(get_python_path "$name")
    local env_name="vllm_env"
    if [[ "$name" == *"Kimi"* ]]; then
        env_name="kimi_env"
    fi

    log "Starting vLLM server for $name on GPU$gpu (port $port) with $env_name..."

    if [ "$DRY_RUN" = true ]; then
        log "[DRY-RUN] Would start: CUDA_VISIBLE_DEVICES=$gpu $python_path -m vllm.entrypoints.openai.api_server --model $model_path --port $port ..."
        echo "dry-run-pid"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=$gpu $python_path -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port "$port" \
        --host "0.0.0.0" \
        --dtype "$VLLM_DTYPE" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        --trust-remote-code \
        $extra_args \
        > "logs/vllm_${name}.log" 2>&1 &

    local pid=$!
    echo $pid > "logs/vllm_${name}.pid"
    ACTIVE_PIDS+=($pid)
    log "Started vLLM server for $name (PID: $pid, Python: $env_name)"

    echo $pid
}

stop_vllm_server() {
    local name="$1"
    local pid_file="logs/vllm_${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            log "Stopping vLLM server for $name (PID: $pid)..."
            kill $pid 2>/dev/null || true
            rm -f "$pid_file"
        fi
    fi
}

stop_all_servers() {
    log "Stopping all vLLM servers..."

    for spec in "${TARGET_MODELS[@]}"; do
        local name url_or_path port_or_key
        parse_model_spec "$spec" name url_or_path port_or_key
        if ! is_remote_api "$spec"; then
            stop_vllm_server "$name"
        fi
    done

    for spec in "${GUIDE_MODELS[@]}"; do
        local name url_or_path port_or_key
        parse_model_spec "$spec" name url_or_path port_or_key
        if ! is_remote_api "$spec"; then
            stop_vllm_server "$name"
        fi
    done
}

cleanup() {
    if [ "$SKIP_CLEANUP" = true ]; then
        log "Skipping cleanup (--skip-cleanup)"
        return
    fi
    stop_all_servers
}

# 生成完整的配置文件
generate_dynamic_config() {
    local output_path="$1"
    local run_id="$2"
    local target_name="$3"
    local target_url="$4"
    local target_api_key="$5"
    local guide_name="$6"
    local guide_url="$7"
    local guide_api_key="$8"

    mkdir -p "$(dirname "$output_path")"

    # 如果 api_key 为空或 "dummy"，则不设置
    local target_api_key_yaml=""
    if [ -n "$target_api_key" ] && [ "$target_api_key" != "dummy" ]; then
        target_api_key_yaml="target_vllm_api_key: \"$target_api_key\""
    fi

    local guide_api_key_yaml=""
    if [ -n "$guide_api_key" ] && [ "$guide_api_key" != "dummy" ]; then
        guide_api_key_yaml="  api_key: \"$guide_api_key\""
    else
        guide_api_key_yaml="  api_key: \"dummy\""
    fi

    cat > "$output_path" << EOF
# 动态实验配置
# Generated at $(date)

run_id: "$run_id"

# Target Model 配置
target_vllm_base_url: "$target_url"
target_vllm_model: "$target_name"
$target_api_key_yaml

# Judge 配置 (远程API)
judge_base_url: "$JUDGE_BASE_URL"
judge_model: "$JUDGE_MODEL"
judge_api_key: "$JUDGE_API_KEY"
judge_mode: "multidim"

beam_width: $BEAM_WIDTH
per_branch_candidates: $PER_BRANCH_CANDIDATES
max_rounds: $MAX_ROUNDS

enable_vision: true
temperature: 0.2
max_tokens: 4096
concurrency: 8

history_strategy: "inherit_parent"
history_memory_key: "history"

# LLM-guided expand策略
expand_strategy: "llm_guided"
llm_guide:
  base_url: "$guide_url"
  model: "$guide_name"
$guide_api_key_yaml
  temperature: 0.7
  max_tokens: 4096
  max_history_rounds: null

enabled_attacks:
  - attacks_strategy.figstep.attack:FigStepAttack
  - attacks_strategy.jood.attack:JOODAttack
  - attacks_strategy.email.attack:EmailThreadAttack
  - attacks_strategy.flowchart.attack:FlowchartAttack
  - attacks_strategy.visual_perturb.jigsaw:JigsawScrambleAttack
  - attacks_strategy.visual_perturb.multimodal_shuffle:MultimodalShuffleAttack
  - attacks_strategy.socialmedia.attack:SlackAttack
  - attacks_strategy.mml.attack:MMLAttack

attack_weights:
  attacks_strategy.figstep.attack:FigStepAttack: 1.0
  attacks_strategy.jood.attack:JOODAttack: 1.0
  attacks_strategy.email.attack:EmailThreadAttack: 1.0
  attacks_strategy.flowchart.attack:FlowchartAttack: 1.0
  attacks_strategy.visual_perturb.jigsaw:JigsawScrambleAttack: 1.0
  attacks_strategy.visual_perturb.multimodal_shuffle:MultimodalShuffleAttack: 1.0
  attacks_strategy.socialmedia.attack:SlackAttack: 1.0
  attacks_strategy.mml.attack:MMLAttack: 1.0

attack_init_kwargs:
  attacks_strategy.figstep.attack:FigStepAttack:
    output_image_dir: runs/$run_id/generated_images
    config:
      font_path: /mnt/disk1/szchen/VLMBenchmark/repo/Jailbreak-Redteam/attacks_strategy/Arial Font/ARIAL.TTF
  attacks_strategy.jood.attack:JOODAttack:
    output_image_dir: runs/$run_id/generated_images
    config:
      harmless_image_path: /mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM/dataset/apple.png
      aug: mixup
      lam: 0.5
  attacks_strategy.email.attack:EmailThreadAttack:
    output_image_dir: runs/$run_id/generated_images
    config:
      llm_base_url: "$JUDGE_BASE_URL"
      llm_api_key: "$JUDGE_API_KEY"
      llm_model_name: "$JUDGE_MODEL"
  attacks_strategy.flowchart.attack:FlowchartAttack:
    output_image_dir: runs/$run_id/generated_images
    config:
      llm_base_url: "$JUDGE_BASE_URL"
      llm_api_key: "$JUDGE_API_KEY"
      llm_model_name: "$JUDGE_MODEL"
  attacks_strategy.visual_perturb.jigsaw:JigsawScrambleAttack:
    output_image_dir: runs/$run_id/generated_images
  attacks_strategy.visual_perturb.multimodal_shuffle:MultimodalShuffleAttack:
    output_image_dir: runs/$run_id/generated_images
    config:
      llm_base_url: "$JUDGE_BASE_URL"
      llm_api_key: "$JUDGE_API_KEY"
      llm_model_name: "$JUDGE_MODEL"
  attacks_strategy.socialmedia.attack:SlackAttack:
    output_image_dir: runs/$run_id/generated_images
    config:
      llm_base_url: "$JUDGE_BASE_URL"
      llm_api_key: "$JUDGE_API_KEY"
      llm_model_name: "$JUDGE_MODEL"
  attacks_strategy.mml.attack:MMLAttack:
    output_image_dir: runs/$run_id/generated_images
    config:
      font_path: /mnt/disk1/szchen/VLMBenchmark/repo/Jailbreak-Redteam/attacks_strategy/Arial Font/ARIAL.TTF
EOF

    log "Generated config: $output_path"
}

# 运行单个实验
run_single_experiment() {
    local target_spec="$1"
    local guide_spec="$2"
    local exp_name="$3"

    local target_name target_path target_port
    local guide_name guide_url guide_api_key

    # 解析 target model
    parse_model_spec "$target_spec" target_name target_path target_port
    local target_url target_api_key
    if is_remote_api "$target_spec"; then
        target_url="$target_path"
        target_api_key="$target_port"
    else
        target_url="http://localhost:$target_port"
        target_api_key="dummy"
    fi

    # 解析 guide model
    parse_model_spec "$guide_spec" guide_name guide_url guide_api_key
    if ! is_remote_api "$guide_spec"; then
        guide_url="http://localhost:$guide_api_key"
        guide_api_key="dummy"
    fi

    log "=========================================="
    log "实验: $exp_name"
    log "=========================================="
    log "Target Model: $target_name"
    log "  URL: $target_url"
    log "Guide Model: $guide_name"
    log "  URL: $guide_url"
    log "Judge Model: $JUDGE_MODEL"
    log "=========================================="

    # 启动需要的服务
    local need_wait=false

    if [ "$DRY_RUN" = false ]; then
        # 启动 Target Model (如果不是远程API)
        if ! is_remote_api "$target_spec"; then
            local pid=$(start_vllm_server "$target_name" "$target_path" "$target_port" "$TARGET_GPU" "")
            if [ -z "$pid" ]; then
                error "Failed to start target model server"
                return 1
            fi
            need_wait=true
        fi

        # 启动 Guide Model (如果不是远程API)
        if ! is_remote_api "$guide_spec"; then
            local pid=$(start_vllm_server "$guide_name" "$guide_url" "$guide_api_key" "$GUIDE_GPU" "")
            if [ -z "$pid" ]; then
                error "Failed to start guide model server"
                return 1
            fi
            need_wait=true
        fi

        # 等待服务就绪
        if [ "$need_wait" = true ]; then
            log ""
            log "Waiting for servers to be ready..."
            if ! is_remote_api "$target_spec"; then
                wait_for_server "http://localhost:$target_port" || return 1
            fi
            if ! is_remote_api "$guide_spec"; then
                local guide_port
                parse_model_spec "$guide_spec" _ _ guide_port
                wait_for_server "http://localhost:$guide_port" || return 1
            fi
        fi
    fi

    # 生成配置并运行实验
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local run_id="${exp_name}-${target_name}-${guide_name}-${timestamp}"
    local temp_config="configs/temp/${exp_name}_${timestamp}.yaml"

    if [ "$DRY_RUN" = true ]; then
        log "[DRY-RUN] Would generate config: $temp_config"
        log "[DRY-RUN] Would run: python scripts/run_batch.py --config $temp_config --dataset $DATASET"
    else
        generate_dynamic_config "$temp_config" "$run_id" \
            "$target_name" "$target_url" "$target_api_key" \
            "$guide_name" "$guide_url" "$guide_api_key"

        python scripts/run_batch.py \
            --config "$temp_config" \
            --dataset "$DATASET" \
            --experiment-name "$run_id" \
            2>&1 | tee "logs/experiment_${run_id}.log"

        local exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            log "Experiment completed successfully!"
        else
            error "Experiment failed with exit code $exit_code"
        fi
    fi

    # 停止本次实验的服务
    if [ "$DRY_RUN" = false ]; then
        if ! is_remote_api "$target_spec"; then
            stop_vllm_server "$target_name"
        fi
        if ! is_remote_api "$guide_spec"; then
            stop_vllm_server "$guide_name"
        fi
    fi

    log ""
}

# ============================================================================
# Exp5: 固定 guide/judge=gpt-4o-mini，遍历所有 target models
# ============================================================================
run_exp5() {
    log "=========================================="
    log "Exp5: 测试不同 Target Models"
    log "固定 Guide/Judge = gpt-4o-mini"
    log "=========================================="

    # 固定使用 gpt-4o-mini 作为 guide (远程API)
    local guide_spec="gpt-4o-mini|https://api.whatai.cc|sk-WE6FQhZoEr4nBz4KTqHuzAYixq6pQG07VZbRhcHaWdGtMTs2"

    local total=${#TARGET_MODELS[@]}
    local current=0

    for target_spec in "${TARGET_MODELS[@]}"; do
        current=$((current + 1))
        local target_name
        parse_model_spec "$target_spec" target_name _ _

        log ""
        log "###############################################"
        log "# Exp5: [$current/$total] Target = $target_name"
        log "###############################################"

        run_single_experiment "$target_spec" "$guide_spec" "exp5-target-comparison" || true
    done

    log ""
    log "Exp5 完成! 共测试 $total 个 Target Models"
}

# ============================================================================
# Exp6: 固定 target=Qwen3-VL-30B-A3B-Instruct，遍历所有 guide models
# ============================================================================
run_exp6() {
    log "=========================================="
    log "Exp6: 测试不同 Guide Models"
    log "固定 Target = $EXP6_FIXED_TARGET"
    log "固定 Judge = gpt-4o-mini"
    log "=========================================="

    # 查找固定的 target model 配置
    local target_spec=""
    for spec in "${TARGET_MODELS[@]}"; do
        local name
        parse_model_spec "$spec" name _ _
        if [ "$name" = "$EXP6_FIXED_TARGET" ]; then
            target_spec="$spec"
            break
        fi
    done

    if [ -z "$target_spec" ]; then
        error "Target model '$EXP6_FIXED_TARGET' not found in TARGET_MODELS"
        exit 1
    fi

    local total=${#GUIDE_MODELS[@]}
    local current=0

    for guide_spec in "${GUIDE_MODELS[@]}"; do
        current=$((current + 1))
        local guide_name
        parse_model_spec "$guide_spec" guide_name _ _

        log ""
        log "###############################################"
        log "# Exp6: [$current/$total] Guide = $guide_name"
        log "###############################################"

        run_single_experiment "$target_spec" "$guide_spec" "exp6-guide-comparison" || true
    done

    log ""
    log "Exp6 完成! 共测试 $total 个 Guide Models"
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    local exp_type=""

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --exp)
                exp_type="$2"
                shift 2
                ;;
            --dataset)
                DATASET="$2"
                shift 2
                ;;
            --skip-cleanup)
                SKIP_CLEANUP=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --list-models)
                list_models
                exit 0
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # 创建日志目录
    mkdir -p logs
    mkdir -p configs/temp

    # 注册清理函数
    trap cleanup EXIT

    case "$exp_type" in
        exp5)
            run_exp5
            ;;
        exp6)
            run_exp6
            ;;
        "")
            error "Please specify experiment type with --exp exp5 or --exp exp6"
            usage
            exit 1
            ;;
        *)
            error "Unknown experiment type: $exp_type. Use 'exp5' or 'exp6'"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
