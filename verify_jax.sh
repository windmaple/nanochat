#!/bin/bash
set -e

echo "--- Verifying Pretrain (base_train.py) ---"
python -m scripts.base_train --num_iterations=5 --device_batch_size=1 --total_batch_size=1 --depth=2 --max_seq_len=128 || echo "Pretrain failed"

echo "--- Verifying Mid-train (mid_train.py) ---"
python -m scripts.mid_train --num_iterations=5 --device_batch_size=1 --total_batch_size=1 --max_seq_len=128 || echo "Mid-train failed"

echo "--- Verifying SFT (chat_sft.py) ---"
python -m scripts.chat_sft --num_iterations=5 --device_batch_size=1 --target_examples_per_step=1 || echo "SFT failed"

echo "--- Verifying RL (chat_rl.py) ---"
python -m scripts.chat_rl --num_epochs=1 --device_batch_size=1 --examples_per_step=1 --num_samples=1 --max_new_tokens=10 || echo "RL failed"
