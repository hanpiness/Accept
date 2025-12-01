#!/bin/bash
nohup accelerate launch inference_scripts/infer_flux_structural_control_val.py \
--pretrained_model_name_or_path= \
--transformer_model_name_or_path=\
--image_variation_model_path=\
--lora_model_path= \
--output_dir=${DATA_OUTPUT_DIR} \
--resolution=1000 --width=1000 --task="wpose" --train_batch_size=1 --val_batch_size=1 --num_train_epochs=20 --checkpointing_steps=1000 \
--learning_rate=1e-4 --lr_warmup_steps=1000 --dataloader_num_workers=8 --rank=128 --kpt_thr=0.0 \
--gradient_accumulation_steps=1 --mixed_precision=bf16 --lr_scheduler="constant" --gradient_checkpointing \
--allow_tf32 --use_time_shift > log.txt 2>&1 &