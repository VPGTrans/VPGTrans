DEVICES=0
N_PROC=1
MASTER_PORT=0
MODEL_SIZE=7b
BATCH_SIZE=16
LLAMA_MODEL=$1

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.run --nproc_per_node=$N_PROC --master_port 1009$MASTER_PORT \
  evaluate.py --cfg-path lavis/projects/blip2/eval/okvqa_llama_eval.yaml \
  --options model.model_type=pretrain_llama"$MODEL_SIZE" model.llama_model=$LLAMA_MODEL \
  run.batch_size_eval=$BATCH_SIZE \
  run.output_dir=llama_eval/okvqa
