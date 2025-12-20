from .base_dataset import BaseDataset
from datasets import load_dataset
import os
import json


class CCVGDataset(BaseDataset):
    def __init__(
            self,
            dataset_path,
            charge_only: bool = False,
            system_prompt: str = "",
    ):
        super().__init__(
            dataset_path,
            charge_only,
            system_prompt
        )

    def load_data(self):
        if not os.path.exists(self.dataset_path):
            if not self.charge_only:
                data = load_dataset("TIM0927/CCVG", split="test")
            else:
                raise ValueError("Charge only dataset must using local file")
        else:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

        for sample in data:
            if self.system_prompt:
                sample["system"] = self.system_prompt

            elif self.charge_only:
                sample["system"] = "你是一个法官，请你根据事实描述预测罪名。"

            sample["label"] = sample["response"]
            del sample["response"]

        return data


if __name__ == '__main__':
    dataset = CCVGDataset(dataset_path="TIM0927/CCVG").data
    for sample in dataset:
        print(sample)
        break
