from dataclasses import dataclass
from typing import Any

from ..engines import HFBackend


@dataclass
class PolicyModel:
    trainer: HFBackend

    @classmethod
    def create(
        cls, train_config: dict[str, Any], model_path: str, use_lora: bool = True
    ) -> "PolicyModel":
        trainer = HFBackend(train_config)
        trainer.load_model(model_path, use_lora=use_lora)
        return cls(trainer=trainer)

    @property
    def model(self) -> Any:
        return self.trainer.model

    @property
    def tokenizer(self) -> Any:
        return self.trainer.tokenizer

    def update(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        return self.trainer.update(trajectories)

    def save(self, output_dir: str) -> None:
        self.trainer.save_model(output_dir)

    def unload(self) -> None:
        self.trainer.unload_model()
