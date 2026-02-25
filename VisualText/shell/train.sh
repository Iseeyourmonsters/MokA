set -x

#export NCCL_IB_GID_INDEX=3
#export NCCL_IB_SL=3
#export NCCL_CHECK_DISABLE=1
#export NCCL_P2P_DISABLE=0
#export NCCL_IB_DISABLE=0
#export NCCL_LL_THRESHOLD=16384
#export NCCL_IB_CUDA_SUPPORT=1
#export NCCL_SOCKET_IFNAME=bond1
#export UCX_NET_DEVICES=bond1
#export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
#export NCCL_COLLNET_ENABLE=0
#export SHARP_COLL_ENABLE_SAT=0
#export NCCL_NET_GDR_LEVEL=2
#export NCCL_IB_QPS_PER_CONNECTION=4
#export NCCL_IB_TC=160
#export NCCL_PXN_DISABLE=0
#export NCCL_NVLS_ENABLE=0
#export NCCL_SOCKET_NTHREADS=4
#export NCCL_NSOCKS_PERTHREAD=4
#export NCCL_IB_TIMEOUT=24
#export NCCL_ASYNC_ERROR_HANDLING=1
#export GLOO_SOCKET_IFNAME=bond1
#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=INFO
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8192
#
#export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400   # 4H waitting
#export TORCH_NCCL_ENABLE_MONITORING=0   # waitting for Infinity
#
#export ACCELERATE_BACKEND=nccl
#export NCCL_TIMEOUT_MINS=30

export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8192
export ACCELERATE_BACKEND=nccl
export TORCH_CUDNN_V8_API_DISABLED=1
export NVIDIA_TF32_OVERRIDE=0
export CUDNN_LOGINFO_DBG=0
export CUDNN_LOGDEST_DBG=0

#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0


export PYTHONPATH="${PYTHONPATH}:$(pwd)"

set -a           # 开启自动导出模式
source ../../.env # 加载 .env 文件（请根据实际相对路径修改）
set +a           # 关闭自动导出模式

wandb login --relogin $WANDB_API_KEY


export WANDB_PROJECT=MOKA_VT
time_stamp=$(date +%Y%m%d)

BASE_RUN_NAME="moka_vl_${time_stamp}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

mkdir -p /tmp/wandb
export WANDB_CACHE_DIR=/tmp/wandb


OUTPUT_DIR=./checkpoints/moka_vt/${BASE_RUN_NAME}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


torchrun --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=29500 \
  train/train.py \
  --output_dir ${OUTPUT_DIR} \
  --fp16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 0.1 \
  --save_total_limit 10 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --do_train True \
  --deepspeed "zero_stage2_config.json" \
  --report_to wandb \
  --run_name $BASE_RUN_NAME \
  --remove_unused_columns False