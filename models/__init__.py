from models.t5_inference import T5Inference

MODELS = {
    "t5": T5Inference,
}


def get_model(name: str, model_dir: str, batch_size: int = 8) -> T5Inference:
    """Returns the model instance based on the selected model type."""
    if name not in MODELS:
        raise ValueError(
            f"Model '{name}' not found. Available options: {list(MODELS.keys())}"
        )

    return MODELS[name](model_dir, batch_size)
