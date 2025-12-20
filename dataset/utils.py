import os.path
from .c3vg import C3VGDataset
from .lcvg import LCVGDataset
from .ccvg import CCVGDataset


def get_dataset(
        dataset_dir,
        dataset_name,
        charge_only: bool = False,
        system_prompt: str = "",
):
    if dataset_name.lower() in ["ccvg"]:
        if charge_only:
            dataset_path = os.path.join(dataset_dir, dataset_name, "wocv_test.jsonl")
        else:
            dataset_path = os.path.join(dataset_dir, dataset_name, "test.jsonl")
        return CCVGDataset(dataset_path=dataset_path, charge_only=charge_only, system_prompt=system_prompt)
    elif dataset_name.lower() in ["c3vg"]:
        dataset_path = os.path.join(dataset_dir, dataset_name, "generation_test.json")
        return C3VGDataset(dataset_path=dataset_path, charge_only=charge_only, system_prompt=system_prompt)
    elif dataset_name.lower() in ["lcvg"]:
        dataset_path = os.path.join(dataset_dir, dataset_name, "test.json")
        return LCVGDataset(dataset_path=dataset_path, charge_only=charge_only, system_prompt=system_prompt)
    else:
        raise ValueError(f"{dataset_name} is not supported.")
