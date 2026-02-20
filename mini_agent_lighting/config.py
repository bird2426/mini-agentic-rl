from typing import Any, Dict, Optional, Tuple, Type

CliConfigurable = Any


class Config:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def lightning_cli(
    *classes: Type[CliConfigurable],
) -> CliConfigurable | Tuple[CliConfigurable, ...]:
    instances = []
    for cls in classes:
        instances.append(cls())
    if len(classes) == 1:
        return instances[0]
    return tuple(instances)


__all__ = ["Config", "lightning_cli", "CliConfigurable"]
