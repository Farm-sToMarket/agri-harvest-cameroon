"""Model definitions: LightGBM, XGBoost, PyTorch YieldNet.

LightGBM uses its native training API (not the sklearn wrapper) for
maximum performance on 10M+ rows.

Torch is imported lazily so this module can be loaded without PyTorch.
"""

from __future__ import annotations

from models.v1.config import LIGHTGBM_PARAMS, XGBOOST_PARAMS, YIELDNET_CONFIG


# ── PyTorch YieldNet ────────────────────────────────────────────────────────

_torch_patched = False


def _ensure_torch_base():
    """Ensure YieldNet inherits from nn.Module (lazy, on first use)."""
    global _torch_patched
    if _torch_patched:
        return
    import torch.nn as nn

    if not issubclass(YieldNet, nn.Module):
        YieldNet.__bases__ = (nn.Module,)
    _torch_patched = True


class YieldNet:
    """Feedforward neural network for tabular yield prediction.

    Architecture: Linear -> ReLU -> BatchNorm -> Dropout (repeated) -> Linear(1)
    """

    def __init__(self, input_dim: int, config: dict | None = None):
        import torch.nn as nn

        _ensure_torch_base()
        nn.Module.__init__(self)

        cfg = config or YIELDNET_CONFIG
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in cfg["hidden_layers"]:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(cfg["dropout"]),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Eagerly patch if torch is already available
try:
    _ensure_torch_base()
except ImportError:
    pass


# ── Parameter builders ──────────────────────────────────────────────────────


def build_lightgbm_params(use_gpu: bool = False) -> dict:
    """Return LightGBM native API training params."""
    params = LIGHTGBM_PARAMS.copy()
    if use_gpu:
        params["device"] = "gpu"
    return params


def build_xgboost_params(use_gpu: bool = False) -> dict:
    """Return XGBoost sklearn-wrapper constructor params."""
    params = XGBOOST_PARAMS.copy()
    if use_gpu:
        params["device"] = "cuda"
    return params
