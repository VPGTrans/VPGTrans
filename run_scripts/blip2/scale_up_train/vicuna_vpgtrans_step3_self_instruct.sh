DEVICES=0 # we use 4 A100-40G in our training.  We use the checkpint_800.pth.
N_PROC=1
MASTER_PORT=1

MODEL_SIZE=7b

BATCH_SIZE=6 # 6 for A100-40G
ACCUM_GRAD_ITERS=$[$[120/BATCH_SIZE]/N_PROC]
echo $ACCUM_GRAD_ITERS

OUTDIR="vl-vicuna7b-step3"
ONLY_PROJ=False

# we use the step2's output
STEP2_CKPT_PATH=$1

VICUNA_PATH=$2

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.run --nproc_per_node=$N_PROC  --master_port 1009$MASTER_PORT \
  train.py --cfg-path lavis/projects/blip2/train/vicuna_vpgtrans_step3_self_instruct.yaml \
  --options model.model_type=pretrain_vicuna"$MODEL_SIZE" model.llama_model=$VICUNA_PATH \
  run.batch_size_train=$BATCH_SIZE \
  run.runner="runner_iter" run.accum_grad_iters=$ACCUM_GRAD_ITERS \
  run.max_iters=800 run.iters_per_inner_epoch=200 \
  run.output_dir="output/BLIP2/$OUTDIR" \
  model.only_proj=$ONLY_PROJ \
  model.pretrained=$STEP2_CKPT_PATH
