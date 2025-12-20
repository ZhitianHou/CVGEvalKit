from .base_dataset import BaseDataset
import os
import json
import re


class LCVGDataset(BaseDataset):
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

    @staticmethod
    def clean_text(text):
        return re.sub(r'\s+', '', text)

    @staticmethod
    def extract_charge(data):
        try:
            return data.get("legal_concept", {}).get("criminal_circumstance", [])[0][0]
        except Exception:
            return ""

    def load_data(self):
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"File path {self.dataset_path} does not exist.")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        new_data = []
        for sample in data:
            fact = self.clean_text(sample.get("fact", ""))
            view = self.clean_text(sample.get("court_view", ""))
            charge = self.extract_charge(sample)

            if not fact or not view or not charge:
                continue  # 跳过缺失数据

            if self.charge_only:
                new_sample = {
                    "system": "你是一个法官，请你根据事实描述预测罪名。",
                    "query": f"按照以下格式输出：<charge>罪名</charge>\n\n事实描述:\n{fact}\nOutput:",
                    "label": f"<charge>{charge}</charge>",
                    "charge": charge
                }
            else:
                new_sample = {
                    "system": "你是一个法官，请你根据事实描述生成法院观点，并根据法院观点预测罪名。",
                    "query": f"按照以下格式输出：<view>法院观点</view>，<charge>罪名</charge>\n\n事实描述:\n{fact}\nOutput:",
                    "label": f"<view>{view}</view>，<charge>{charge}</charge>",
                    "charge": charge
                }
            new_data.append(new_sample)
        data = new_data

        return data


if __name__ == '__main__':
    dataset = LCVGDataset(dataset_path="../data/LCVG/test.json").data
    for sample in dataset:
        print(sample)
        break
