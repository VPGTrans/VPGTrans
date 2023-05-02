
img2dataset --url_list "./meta" --input_format "parquet" --url_col "URL" \
  --caption_col "TEXT" --output_format webdataset --output_folder laion400m-data \
  --processes_count 16 --thread_count 128 --image_size 256 \
  --save_additional_columns '["all_captions","all_similarities"]' \
  --enable_wandb True --resize_mode "keep_ratio" \
  --resize_only_if_bigger \
  --number_sample_per_shard 80000