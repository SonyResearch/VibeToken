# Run training with 8 GPUs across 2 nodes (4 GPUs per node)
NODE_RANK=${RANK:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} 
MASTER_PORT=${MASTER_PORT:-9871}

accelerate launch \
    --num_machines=1 \
    --num_processes=8 \
    --machine_rank=$NODE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --same_network \
    scripts/train_tokenvibe.py \
    config=configs/training/VibeToken_small.yaml