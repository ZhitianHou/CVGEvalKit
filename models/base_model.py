from swift.llm import (
    PtEngine, RequestConfig, VllmEngine
)


def get_engine(
        model_path: str = "",
        lora_checkpoint: str = "",
        engine_kwargs: dict = None,
        request_kwargs: dict = None,
):
    infer_backbone = engine_kwargs.pop("infer_backbone", "vllm")
    if infer_backbone == "vllm":
        engine = VllmEngine(model_path, adapters=lora_checkpoint, **engine_kwargs)
    else:
        engine = PtEngine(model_path, adapters=lora_checkpoint, **engine_kwargs)
    request_config = RequestConfig(**request_kwargs)

    return engine, request_config
