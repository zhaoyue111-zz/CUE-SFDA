from __future__ import annotations

import gc
import hashlib
import json
import pydoc
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from cue_sfda.modeling.lora import (
    freeze_except_lora,
    inject_decoder_qkv_lora,
    inject_linear_lora,
)


_TORCHVISION_NMS_LIBRARY = None


class DynamicVoxTell(nn.Module):
    """VoxTell forward pass with positional encodings derived from patch size."""

    def __init__(
        self,
        source_model: nn.Module,
        native_patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.source_model = source_model
        self.native_patch_size = tuple(int(value) for value in native_patch_size)

    def forward(
        self,
        image: torch.Tensor,
        text_embedding: torch.Tensor,
        return_features: bool = False,
        return_decoder_outputs: bool = False,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        if tuple(image.shape[-3:]) == self.native_patch_size:
            if return_features:
                prediction, diagnostics = self.source_model(
                    image,
                    text_embedding,
                    return_features=True,
                )
                features = diagnostics["d5_pre_head"]
                if features.shape[1] != 1:
                    raise ValueError(
                        "Training expects exactly one text prompt feature map"
                    )
                return prediction, features[:, 0]
            if return_decoder_outputs:
                return self.source_model(
                    image,
                    text_embedding,
                    return_decoder_outputs=True,
                )
            return self.source_model(image, text_embedding)

        if return_decoder_outputs:
            raise ValueError(
                "Decoder-stage diagnostics require the native VoxTell patch size"
            )

        from positional_encodings.torch_encodings import PositionalEncoding3D

        model = self.source_model
        skips = model.encoder(image)
        selected = skips[model.selected_decoder_layer]
        memory = rearrange(selected, "b c d h w -> b h w d c")
        memory = model.project_bottleneck_embed(memory)
        batch, height, width, depth, channels = memory.shape
        position_module = PositionalEncoding3D(model.query_dim).to(memory.device)
        position_seed = torch.zeros(
            1,
            height,
            width,
            depth,
            channels,
            device=memory.device,
            dtype=memory.dtype,
        )
        position = position_module(position_seed)
        position = rearrange(position, "b h w d c -> (h w d) b c")
        memory = rearrange(memory, "b h w d c -> (h w d) b c")
        text = text_embedding
        if text.ndim == 4 and text.shape[2] == 1:
            text = text.squeeze(2)
        if text.ndim != 3:
            raise ValueError(
                f"Expected text embedding [B,N,C], got {tuple(text.shape)}"
            )
        if text.shape[0] == 1 and batch > 1:
            text = text.expand(batch, -1, -1)
        text = rearrange(text, "b n c -> n b c")
        text = model.project_text_embed(text)
        mask_embedding, _ = model.transformer_decoder(
            tgt=text,
            memory=memory,
            pos=position,
            memory_key_padding_mask=None,
        )
        mask_embedding = rearrange(mask_embedding, "n b c -> b n c")
        mask_embeddings = [
            projection(mask_embedding)
            for projection in model.project_to_decoder_channels
        ]
        result = model.decoder(
            skips,
            mask_embeddings,
            return_features=return_features,
        )
        if return_features:
            outputs, features = result
            return outputs[0], features
        return result[0]


@dataclass
class VoxTellRuntime:
    model: DynamicVoxTell
    embedding: torch.Tensor
    trainable_parameters: list[nn.Parameter]
    patch_size: tuple[int, int, int]
    device: torch.device
    lora_modules: int


def _add_voxtell_path(voxtell_root: str | Path) -> Path:
    root = Path(voxtell_root).expanduser().resolve()
    if not (root / "voxtell").is_dir():
        raise FileNotFoundError(f"Invalid VoxTell root: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _prompt_cache_path(cache_dir: Path, prompt: str, text_encoder: Path) -> Path:
    payload = f"{prompt}\n{text_encoder.resolve()}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    slug = "".join(character if character.isalnum() else "_" for character in prompt)
    slug = slug.strip("_")
    return cache_dir / f"{slug or 'prompt'}_{digest}.pt"


def _encode_prompt(
    prompt: str,
    text_encoder: Path,
    cache_dir: Path,
    device: torch.device,
    dtype_name: str | None = None,
) -> torch.Tensor:
    cache_path = _prompt_cache_path(cache_dir, prompt, text_encoder)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        embedding = payload["embedding"] if isinstance(payload, dict) else payload
        return embedding.float()
    # Transformers 4.57 imports torchvision while resolving Qwen3Model.
    # Some mixed CUDA installs have a working torch but a torchvision wheel
    # without the compiled torchvision::nms operator. Text embedding does not
    # use torchvision, so register only the missing schema and let the import
    # continue. Keep the Library alive for the process lifetime.
    try:
        import torchvision  # noqa: F401
    except Exception:
        for module_name in list(sys.modules):
            if module_name == "torchvision" or module_name.startswith("torchvision."):
                sys.modules.pop(module_name, None)
        try:
            global _TORCHVISION_NMS_LIBRARY
            _TORCHVISION_NMS_LIBRARY = torch.library.Library("torchvision", "DEF")
            _TORCHVISION_NMS_LIBRARY.define(
                "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
            )
        except Exception:
            pass
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction

    tokenizer = AutoTokenizer.from_pretrained(
        str(text_encoder),
        padding_side="left",
        local_files_only=True,
    )
    dtype = (
        getattr(torch, str(dtype_name))
        if dtype_name is not None
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    try:
        backbone = AutoModel.from_pretrained(
            str(text_encoder),
            local_files_only=True,
            torch_dtype=dtype,
        )
    except (ValueError, KeyError):
        # The local Qwen3-Embedding checkpoint declares Qwen3ForCausalLM.
        # Older Transformers mappings may not expose its base Qwen3Model via
        # AutoModel, while AutoModelForCausalLM supports the saved architecture.
        backbone = AutoModelForCausalLM.from_pretrained(
            str(text_encoder),
            local_files_only=True,
            torch_dtype=dtype,
        )
    backbone = backbone.eval().to(device)
    tokens = tokenizer(
        wrap_with_instruction([prompt]),
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    tokens = {name: value.to(device) for name, value in tokens.items()}
    with torch.inference_mode():
        outputs = backbone(**tokens, output_hidden_states=True)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        embedding = last_token_pool(hidden, tokens["attention_mask"])
        embedding = embedding.reshape(1, 1, -1).float().cpu()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt": prompt, "embedding": embedding}, cache_path)
    del backbone
    del tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embedding


def _build_source_model(model_dir: Path) -> tuple[nn.Module, tuple[int, int, int]]:
    from voxtell.model.voxtell_model import VoxTellModel

    plans_path = model_dir / "plans.json"
    checkpoint_path = model_dir / "checkpoint_final.pth"
    with plans_path.open("r", encoding="utf-8") as handle:
        plans = json.load(handle)
    configuration = plans["configurations"]["3d_fullres"]
    architecture = configuration["architecture"]
    kwargs = dict(architecture["arch_kwargs"])
    for key in architecture["_kw_requires_import"]:
        if kwargs[key] is not None:
            kwargs[key] = pydoc.locate(kwargs[key])
    model = VoxTellModel(
        input_channels=1,
        **kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["network_weights"])
    patch_size = tuple(int(value) for value in configuration["patch_size"])
    return model, patch_size


def load_prompt_embedding(
    config: dict,
    prompt: str,
    cache_dir: str | Path,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load or encode one fixed prompt without rebuilding the segmentation model."""
    model_config = config["model"]
    _add_voxtell_path(model_config["voxtell_root"])
    if device is None:
        gpu_id = config["experiment"].get("gpu_id")
        device = torch.device(
            model_config.get("device", "cuda:0")
            if gpu_id is None
            else f"cuda:{int(gpu_id)}"
        )
    text_encoder = Path(model_config["text_encoder"]).expanduser().resolve()
    embedding = _encode_prompt(
        str(prompt),
        text_encoder,
        Path(cache_dir),
        device,
        model_config.get("text_encoder_dtype"),
    )
    return embedding.to(device)


def build_runtime(config: dict, cache_dir: str | Path) -> VoxTellRuntime:
    model_config = config["model"]
    _add_voxtell_path(model_config["voxtell_root"])
    gpu_id = config["experiment"].get("gpu_id")
    if gpu_id is None:
        requested_device = torch.device(model_config.get("device", "cuda:0"))
    else:
        requested_device = torch.device(f"cuda:{int(gpu_id)}")
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested_device}")
    model_dir = Path(model_config["model_dir"]).expanduser().resolve()
    prompt = str(model_config["prompt"])
    embedding = load_prompt_embedding(
        config,
        prompt,
        cache_dir,
        requested_device,
    )
    source_model, source_patch_size = _build_source_model(model_dir)
    configured_patch_size = tuple(
        int(value) for value in config["data"]["patch_size"]
    )
    if configured_patch_size != source_patch_size:
        raise ValueError(
            "data.patch_size must match the VoxTell checkpoint patch size for "
            "baseline-equivalent inference: "
            f"configured={configured_patch_size}, checkpoint={source_patch_size}"
        )
    dynamic_model = DynamicVoxTell(source_model, source_patch_size)
    count = inject_decoder_qkv_lora(
        source_model.transformer_decoder,
        rank=int(model_config["lora_rank"]),
        alpha=float(model_config["lora_alpha"]),
        dropout=float(model_config.get("lora_dropout", 0.0)),
    )
    linear_targets = tuple(model_config.get("lora_linear_targets", ()))
    if "text_projection" in linear_targets:
        count += inject_linear_lora(
            source_model.project_text_embed,
            rank=int(model_config["lora_rank"]),
            alpha=float(model_config["lora_alpha"]),
        )
    if "final_mask_projection" in linear_targets:
        count += inject_linear_lora(
            source_model.project_to_decoder_channels[0],
            rank=int(model_config["lora_rank"]),
            alpha=float(model_config["lora_alpha"]),
        )
    parameters = freeze_except_lora(dynamic_model)
    dynamic_model.to(requested_device)
    embedding = embedding.to(requested_device)
    return VoxTellRuntime(
        model=dynamic_model,
        embedding=embedding,
        trainable_parameters=parameters,
        patch_size=source_patch_size,
        device=requested_device,
        lora_modules=count,
    )
