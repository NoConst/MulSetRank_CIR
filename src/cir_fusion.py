import json
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

FUSION_STATE_FILENAME = "cir_fusion.pt"
FUSION_CONFIG_FILENAME = "cir_fusion_config.json"


class AdaptiveResidualFusion(nn.Module):
    """
    CIR-oriented fusion head.

    Starts from the stable CLIP baseline (image + text) and learns a gated
    residual update from text-conditioned image modulation plus pairwise
    feature interactions.
    """

    fusion_type = "adaptive_residual"

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim or max(256, embedding_dim // 2))
        self.dropout = float(dropout)

        pair_dim = self.embedding_dim * 4
        self.pair_norm = nn.LayerNorm(pair_dim)

        self.film_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim * 2),
        )
        self.interaction_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )

        self.output_norm = nn.LayerNorm(self.embedding_dim)
        self.image_residual_weight = nn.Parameter(torch.tensor(1.0))
        self.text_residual_weight = nn.Parameter(torch.tensor(1.0))
        self.update_scale = nn.Parameter(torch.tensor(1.0))

        self.reset_parameters()

    def reset_parameters(self):
        # Zero-init the update branches so training starts from image + text.
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
        nn.init.zeros_(self.interaction_mlp[-1].weight)
        nn.init.zeros_(self.interaction_mlp[-1].bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

    def get_config(self) -> dict:
        return {
            "fusion_type": self.fusion_type,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        module_param = next(self.parameters())
        compute_device = module_param.device
        compute_dtype = module_param.dtype

        image_features = F.normalize(
            image_features.to(device=compute_device, dtype=compute_dtype),
            dim=-1,
        )
        text_features = F.normalize(
            text_features.to(device=compute_device, dtype=compute_dtype),
            dim=-1,
        )

        pair_features = torch.cat(
            [
                image_features,
                text_features,
                image_features * text_features,
                image_features - text_features,
            ],
            dim=-1,
        )
        pair_features = self.pair_norm(pair_features)

        film_scale, film_shift = self.film_mlp(pair_features).chunk(2, dim=-1)
        film_delta = torch.tanh(film_scale) * image_features + film_shift
        interaction_delta = self.interaction_mlp(pair_features)
        edit_gate = torch.sigmoid(self.gate_mlp(pair_features))

        base = self.image_residual_weight * image_features + self.text_residual_weight * text_features
        fused = base + self.update_scale * edit_gate * (film_delta + interaction_delta)
        return F.normalize(self.output_norm(fused), dim=-1)


def build_cir_fusion_module(
    fusion_type: str,
    embedding_dim: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
) -> Optional[nn.Module]:
    fusion_type = fusion_type.lower()
    if fusion_type == "sum":
        return None
    if fusion_type == "adaptive_residual":
        return AdaptiveResidualFusion(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported fusion_type: {fusion_type}")


def get_model_cir_fusion(model) -> Optional[nn.Module]:
    candidates = [
        model,
        getattr(model, "module", None),
        getattr(model, "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        fusion = getattr(candidate, "cir_fusion", None)
        if isinstance(fusion, nn.Module):
            return fusion
    return None


def attach_cir_fusion(
    model,
    embedding_dim: int,
    fusion_type: str = "adaptive_residual",
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
    log: Optional[logging.Logger] = None,
) -> Optional[nn.Module]:
    fusion_module = build_cir_fusion_module(
        fusion_type=fusion_type,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.cir_fusion = fusion_module

    if log is not None:
        if fusion_module is None:
            log.info("Using baseline CIR fusion: normalized image + text sum.")
        else:
            log.info(
                "Attached CIR fusion head: "
                f"{fusion_module.fusion_type} "
                f"(hidden_dim={fusion_module.hidden_dim}, dropout={fusion_module.dropout})"
            )
    return fusion_module


def compose_query_features(
    model,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    fusion_module = get_model_cir_fusion(model)
    if fusion_module is None:
        return F.normalize(image_features + text_features, dim=-1)
    return fusion_module(image_features, text_features)


def save_cir_fusion(
    model,
    save_dir: Union[str, Path],
    log: Optional[logging.Logger] = None,
) -> bool:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fusion_module = get_model_cir_fusion(model)
    if fusion_module is None:
        config = {"fusion_type": "sum"}
        with open(save_dir / FUSION_CONFIG_FILENAME, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, sort_keys=True)
        return False

    config = fusion_module.get_config()
    with open(save_dir / FUSION_CONFIG_FILENAME, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    torch.save(
        {
            "config": config,
            "state_dict": fusion_module.state_dict(),
        },
        save_dir / FUSION_STATE_FILENAME,
    )
    if log is not None:
        log.info(f"Saved CIR fusion head to {save_dir / FUSION_STATE_FILENAME}")
    return True


def load_cir_fusion(
    model,
    load_dir: Union[str, Path],
    log: Optional[logging.Logger] = None,
) -> bool:
    load_dir = Path(load_dir)
    config_path = load_dir / FUSION_CONFIG_FILENAME
    if not config_path.exists():
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    fusion_type = config.get("fusion_type", "sum")
    embedding_dim = config.get("embedding_dim")
    if fusion_type != "sum" and embedding_dim is None:
        raise ValueError(f"Missing embedding_dim in {config_path}")

    attach_cir_fusion(
        model=model,
        embedding_dim=embedding_dim or 1,
        fusion_type=fusion_type,
        hidden_dim=config.get("hidden_dim"),
        dropout=config.get("dropout", 0.1),
        log=None,
    )

    fusion_module = get_model_cir_fusion(model)
    state_path = load_dir / FUSION_STATE_FILENAME
    if fusion_module is not None and state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        fusion_module.load_state_dict(state["state_dict"], strict=True)

    if log is not None:
        log.info(f"Loaded CIR fusion config from {config_path}")
    return True
