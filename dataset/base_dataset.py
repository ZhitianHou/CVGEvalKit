class BaseDataset:
    def __init__(
            self,
            dataset_path,
            charge_only: bool = False,
            system_prompt: str = "",
    ):
        self.dataset_path = dataset_path
        self.charge_only = charge_only
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = "你是一个法官，请你根据事实描述生成法院观点，并根据法院观点预测罪名。"

        self.data = self.load_data()

    def load_data(self):
        pass
