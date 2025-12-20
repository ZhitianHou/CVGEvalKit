import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest, VllmEngine
)
from swift.tuners import Swift
import json
from tqdm import tqdm
from utils.metrics import evaluation
import argparse
from dataset.utils import get_dataset
from models.base_model import get_engine


def main(
        model_path,
        data,
        output_path,
        lora_checkpoint=None,
        batch_size: int = 1,
        reasoning: bool = False,
        request_kwargs: dict = None,
        engine_kwargs: dict = None,
):
    # set key
    system_key = "system"
    query_key = "query"
    label_key = "label"
    charge_key = "charge"

    # get engine and request config
    engine, request_config = get_engine(
        model_path=model_path,
        lora_checkpoint=lora_checkpoint,
        engine_kwargs=engine_kwargs,
        request_kwargs=request_kwargs,
    )
    if not reasoning:
        suffix = "/no_think" if model_path.split("/")[-1].split("-")[0] == "Qwen3" and lora_checkpoint is None else ""
    else:
        suffix = ""

    # inference with batches
    results = []
    for i in tqdm(range(0, len(data), batch_size), desc="Generation"):
        batch_data = data[i:i+batch_size]

        # batch inference
        infer_requests = []
        for obj in batch_data:
            infer_requests.append(
                InferRequest(
                    messages=[
                        {'role': 'system', 'content': obj[system_key]},
                        {'role': 'user', 'content': obj[query_key].replace(" ", "")+suffix}
                    ]
                )
            )

        # data preview
        if i == 0:
            print("="*10, "Data Preview", "="*10)
            print(infer_requests[0].messages)

        # get response
        responses = engine.infer(infer_requests, request_config)
        # print(f'response0: {responses[0].choices[0].message.content}')

        # organize results
        for obj, resp in zip(batch_data, responses):
            results.append({
                "system": obj[system_key],
                "query": obj[query_key].replace(" ", ""),
                "response": resp.choices[0].message.content,
                "label": obj[label_key].replace(" ", ""),
                "charge": obj[charge_key]
            })

    # save as jsonl
    with open(output_path, "w", encoding="utf-8") as outfile:
        for item in results:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"Generation finished, results saved to {output_path}")


def get_parser():
    parser = argparse.ArgumentParser(description="CVGEvalKit")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory of models")
    parser.add_argument("--model_paths", type=str, required=True,
                        help="List of models paths")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory of dataset")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="List of dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory of output")
    parser.add_argument("--output_paths", type=str, required=True,
                        help="Paths of output of each model")
    parser.add_argument("--charge_only", type=str, default="false",
                        help="Generating charge only or not")
    parser.add_argument("--lora_paths", type=str, default=None,
                        help="Paths of LoRA checkpoint")
    parser.add_argument("--use_vllm", type=str, default="true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--default_system", type=str, default="")
    parser.add_argument("--reasoning", type=str, default="false")
    parser.add_argument("--infer_backbone", type=str, default="vllm")
    parser.add_argument("--use_async_engine", type=str, default="true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--enforce_eager", type=str, default="true")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # get model paths and output paths
    model_paths = args.model_paths.split(",")
    output_paths = args.output_paths.split(",")
    assert len(model_paths) == len(output_paths), "Number of models must equal with number of output paths"

    # get lora paths
    if args.lora_paths:
        lora_paths = args.lora_paths.split(",")
        assert len(model_paths) == len(lora_paths), "Number of models must equal with number of lora checkpoints"
    else:
        lora_paths = [None for _ in range(len(model_paths))]

    # set request kwargs
    request_kwargs = {
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning": args.reasoning == "true",
    }

    # set engine kwargs
    engine_kwargs = None
    if args.use_vllm == "true":
        os.environ['VLLM_WORKER_MULTIPROC_METHOD']="spawn"
        engine_kwargs = {
            "infer_backbone": args.infer_backbone,
            "use_async_engine": args.use_async_engine == "true",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager == "true"
        }

    # set model and output mapping
    model_dir = args.model_dir
    model_map = {
        model_n: output_p for model_n, output_p in zip(model_paths, output_paths)
    }
    print(f"Model and Output Mapping:\n{model_map}")

    # set other args
    output_dir = args.output_dir
    charge_only = args.charge_only == "true"

    # set dataset
    dataset_names = args.dataset_name.split(",")
    all_dataset = {
        dn: get_dataset(
            dataset_dir=args.dataset_dir,
            dataset_name=dn,
            charge_only=charge_only,
            system_prompt=args.default_system,
        ).data for dn in dataset_names
    }

    # set inference setting
    batch_size = request_kwargs.pop("batch_size", 1)
    reasoning = request_kwargs.pop("reasoning", False)

    # print kwargs
    print(f"Request Kwargs: {request_kwargs}\nEngine Kwargs: {engine_kwargs}\n"
          f"Output Dir: {output_dir}\nCharge Only: {charge_only}")

    # evaluating each dataset with each model
    for (model_n, output_p), lora_p in zip(model_map.items(), lora_paths):
        print(f"Evaluating Model: {model_n}, LoRA: {lora_p} ...")

        os.makedirs(os.path.join(output_dir, output_p), exist_ok=True)
        metrics_path = os.path.join(output_dir, output_p, "metrics.txt")
        for dn, dataset in all_dataset.items():
            print(f"Evaluating {dn} ... ")
            result_path = os.path.join(output_dir, output_p, dn + ".jsonl")
            if not os.path.exists(result_path):
                main(
                    model_path=os.path.join(model_dir, model_n),
                    data=dataset,
                    output_path=result_path,
                    lora_checkpoint=lora_p,
                    batch_size=batch_size,
                    reasoning=reasoning,
                    request_kwargs=request_kwargs,
                    engine_kwargs=engine_kwargs,
                )
            else:
                print(f"Load results from: {result_path} ...")
            evaluation(dn, result_path, metrics_path, charge_only=charge_only)
