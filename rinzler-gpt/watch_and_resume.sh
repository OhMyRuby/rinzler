#!/usr/bin/env bash
set -e

LOG=/home/matrix9180/Projects/rinzler/rinzler-gpt/training.log
RESUME_LOG=/home/matrix9180/Projects/rinzler/rinzler-gpt/training_resume.log
CKPT=/home/matrix9180/Projects/rinzler/rinzler-gpt/checkpoint.json
DIR=/home/matrix9180/Projects/rinzler/rinzler-gpt

echo "Watching for completion..."

while ! grep -q "Training complete" "$LOG"; do
  sleep 15
done

echo "Run complete. Starting resume..."

cd "$DIR"
OMP_NUM_THREADS=4 bundle exec ruby train.rb \
  --resume "$CKPT" \
  --steps 31800 \
  --batch-size 8 \
  --eval-every 100 \
  --gen-every 1000 \
  --gen-len 400 \
  --save-every 500 \
  2>&1 | tee "$RESUME_LOG"
