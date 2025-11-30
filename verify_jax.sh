#!/bin/bash
set -e

echo "--- Verifying Pretrain (base_train.py) ---"
if python -m scripts.base_train --num_iterations=1 --device_batch_size=1 --total_batch_size=128 --depth=2 --max_seq_len=128; then
  echo "Pretrain successful"
else
  echo "Pretrain failed"
  exit 1
fi

echo "--- Verifying Mid-train (mid_train.py) ---"
if python -m scripts.mid_train --num_iterations=1  --device_batch_size=1 --total_batch_size=128 --max_seq_len=128; then
  echo "Mid-train successful"
else
  echo "Mid-train failed"
  exit 1
fi

echo "--- Verifying SFT (chat_sft.py) ---"
if python -m scripts.chat_sft --num_iterations=1 --device_batch_size=1 --target_examples_per_step=1 > /dev/null 2>&1; then
  echo "SFT successful"
else
  echo "SFT failed"
  exit 1
fi

echo "--- Verifying RL (chat_rl.py) ---"
if python -m scripts.chat_rl --num_epochs=1 --device_batch_size=1 --run="dummy"; then
  echo "RL successful"
else
  echo "RL failed"
  exit 1
fi
