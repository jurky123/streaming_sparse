#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if tmux has-session -t opt20_train 2>/dev/null; then
  echo 'Opt20 后台会话已存在；不会重复启动。'
else
  tmux new-session -d -s opt20_train -c "$PWD" 'exec streaming_sparse/.venv/bin/python -u -m migration.unified_router.opt20.workflow >> streaming_sparse/v10/experiments/opt20_all29_tile_finetune_20261025/workflow.log 2>&1'
  echo 'Opt20 已在独立 tmux 会话启动；自动续训并接续视频评估。'
fi
echo '查看进度：bash opt20_status.sh --watch'
