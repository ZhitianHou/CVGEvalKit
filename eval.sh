HOME=/path/to/your/home
export PYTHONPATH=$HOME:$PYTHONPATH

nohup python -u eval.py \
      --model_dir "/directory/to/your/model/parent/directory" \
      --model_paths "/directory/to/your/specific/model" \
      --dataset_dir "data" \
      --dataset_name "CCVG,C3VG,LCVG" \
      --output_dir "eval_results" \
      --output_paths "/output/path" \
      --charge_only "false" \
      --use_vllm "true" \
      --batch_size 16 \
      --temperature 0.0 \
      --max_tokens 512 \
      --default_system "" \
      --reasoning "false" \
      --infer_backbone "vllm" \
      --use_async_engine "true" \
      --gpu_memory_utilization 0.7 \
      --enforce_eager "true" 2>&1 | tee log.txt
