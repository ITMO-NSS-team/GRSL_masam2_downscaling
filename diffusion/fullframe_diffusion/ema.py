from __future__ import annotations

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def model_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name] = value.detach().clone()
        return state

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {
                name: value.detach().cpu() for name, value in self.shadow.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        loaded = state["shadow"]
        if loaded.keys() != self.shadow.keys():
            raise ValueError("EMA parameter names do not match the model")
        for name in self.shadow:
            self.shadow[name].copy_(loaded[name].to(self.shadow[name].device))
