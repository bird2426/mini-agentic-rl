from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_path: str
    use_lora: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class GRPOConfig:
    samples_per_prompt: int = 4
    prompt_batch_size: int = 8
    kl_coef: float = 0.0


@dataclass
class TrainConfig:
    total_epochs: int = 1
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 8
    max_samples: int | None = None
