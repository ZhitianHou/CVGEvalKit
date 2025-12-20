# CVGEvalKit

This is the repository of CVGEvalKit in *Exploring Lightweight Large Language Models for Court View Generation*

## Quick Start
1. Download this repository
```bash
git clone git@github.com:ZhitianHou/CVGEvalKit.git
cd CVGEvalKit
```

2. Prepare the Environment
```bash
conda create -n cvgevalkit python==3.10
conda activate cvgevalkit
pip install -r requirements.txt
```

3. Download the Datasets

Download each data from link below.

| Dataset | Link                                                                                                                          |
|---------|-------------------------------------------------------------------------------------------------------------------------------|
| C3VG    | [Google Drive](https://drive.google.com/file/d/1LzLpqe3YJwtQG8i3RBB49Gkx3Mpd_38m/view)                                        |
| LCVG    | [Baidu Netdisk](https://pan.baidu.com/share/init?surl=GsdoAVcd7KavY3Tz7SHZyA&pwd=g2zd)                                        |
| CCVG    | [Huggingface](https://huggingface.co/datasets/TIM0927/CCVG), [ModelScope](https://www.modelscope.cn/datasets/ZhitianHou/CCVG) |

4. Run Eval

```bash
bash eval.sh
```

Example of eval.sh
```bash
HOME=/path/to/your/home
export PYTHONPATH=$HOME:$PYTHONPATH

nohup python -u eval.py \
      --model_dir "/home/user/models" \
      --model_paths "internlm2_5-1_8b-chat" \
      --dataset_dir "data" \
      --dataset_name "CCVG,C3VG,LCVG" \
      --output_dir "eval_results" \
      --output_paths "./results/internlm2_5-1_8b-chat" \
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

```

## Citation
If you find this project helpful, please consider citing our paper:

```bibtex
coming soon
```