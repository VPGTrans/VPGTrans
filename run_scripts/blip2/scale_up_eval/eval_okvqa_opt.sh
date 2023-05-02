DEVICES=$1
N_PROC=$2
MASTER_PORT=$3
MODEL_SIZE=$4
PRETRAINED_DIR=$5
CKPT="$PRETRAINED_DIR"/"$6"
BATCH_SIZE=$7

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.run --nproc_per_node=$N_PROC --master_port 1009$MASTER_PORT \
  evaluate.py --cfg-path lavis/projects/blip2/eval/okvqa_opt_eval.yaml \
  --options model.model_type=pretrain_opt"$MODEL_SIZE" model.opt_model=facebook/opt-"$MODEL_SIZE" \
  model.pretrained=$CKPT \
  run.batch_size_eval=$BATCH_SIZE \
  run.output_dir="$PRETRAINED_DIR"/trans/$6/okvqa
