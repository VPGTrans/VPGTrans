DEVICES=0
N_PROC=1
MASTER_PORT=1

MODEL_SIZE=7b

BATCH_SIZE=16 # 36 for A100-40G
ACCUM_GRAD_ITERS=$[$[1728/BATCH_SIZE]/N_PROC]
echo $ACCUM_GRAD_ITERS

OUTDIR="vl-llama7b-step1"
ONLY_PROJ=True

# we use blip2_pretrained_opt6.7b.pth
# from https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt6.7b.pth
# please download in advance
QFORMER_PATH=$1

# if you have an initialization, use the intialization
# if you do not have, set it to null string "".
PROJ_PATH=$2

LLAMA_PATH=$3

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.run --nproc_per_node=$N_PROC  --master_port 1009$MASTER_PORT \
  train.py --cfg-path lavis/projects/blip2/train/llama_vpgtrans_step1_proj_warmup.yaml \
  --options model.model_type=pretrain_llama"$MODEL_SIZE" model.llama_model=$LLAMA_PATH \
  run.warmup_steps=0 run.accum_grad_iters=$ACCUM_GRAD_ITERS run.batch_size_train=$BATCH_SIZE \
  run.output_dir="output/BLIP2/$OUTDIR" \
  model.only_proj=$ONLY_PROJ \
  model.qformer_weight_path=$QFORMER_PATH \
  model.proj_weight_path=$PROJ_PATH run.min_lr=5e-4 run.init_lr=5e-4 run.max_epoch=1

