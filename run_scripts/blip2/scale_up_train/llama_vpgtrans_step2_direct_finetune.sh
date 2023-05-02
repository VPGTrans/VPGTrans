DEVICES=0 # we use 8 A100-40G in our training.  We use the checkpint_45000.pth.
N_PROC=1
MASTER_PORT=1

MODEL_SIZE=7b

BATCH_SIZE=16 # 36 for A100-40G
ACCUM_GRAD_ITERS=$[$[1728/BATCH_SIZE]/N_PROC]
echo $ACCUM_GRAD_ITERS

OUTDIR="vl-llama7b-step2"
ONLY_PROJ=False

# we use blip2_pretrained_opt6.7b.pth
# from https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt6.7b.pth
# please download in advance
QFORMER_PATH=$1

# Please set to step1's output
PROJ_PATH=$2

LLAMA_PATH=$3

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.run --nproc_per_node=$N_PROC  --master_port 1009$MASTER_PORT \
  train.py --cfg-path lavis/projects/blip2/train/llama_vpgtrans_step2_direct_finetune.yaml \
  --options model.model_type=pretrain_llama"$MODEL_SIZE" model.llama_model=$LLAMA_PATH \
  run.warmup_steps=0 run.accum_grad_iters=$ACCUM_GRAD_ITERS run.batch_size_train=$BATCH_SIZE \
  run.min_lr=5e-5 run.runner="runner_iter" \
  run.max_iters=50000 run.iters_per_inner_epoch=5000 \
  run.output_dir="output/BLIP2/$OUTDIR" \
  model.only_proj=$ONLY_PROJ \
  model.qformer_weight_path=$QFORMER_PATH \
  model.proj_weight_path=$PROJ_PATH
