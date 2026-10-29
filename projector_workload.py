import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jmp
import numpy as np
import optax
import wandb
from experiments.multimodality.base.model import Vlm, VlmConfig, load_pretrained
from experiments.multimodality.base.train import caption_batch, compute_loss
from huggingface_hub import snapshot_download
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import RepoRef
from levanter.distributed import DistributedConfig
from levanter.layers.attention import AttentionBackend
from levanter.models.qwen import Qwen3Config
from levanter.models.siglip import SiglipVisionConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import trainables_only
from levanter.utils.tree_utils import inference_mode
from PIL import Image, ImageDraw
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast, SiglipImageProcessor


def checksum(component):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(eqx.filter(component, eqx.is_array)):
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def parameter_count(component):
    return sum(leaf.size for leaf in jax.tree.leaves(eqx.filter(component, eqx.is_array)))


def cached_checkpoint(reference: str, cache: Path) -> str:
    ref = RepoRef.from_string(reference)
    return snapshot_download(
        ref.model_name_or_path,
        revision=ref.revision,
        cache_dir=str(cache),
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer*",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "vocab*",
            "merges.txt",
        ],
    )


def run(output_root, steps=20, tiny=False, wandb_project="", wandb_entity="", run_prefix=""):
    if wandb_project and not run_prefix:
        raise ValueError("--run-prefix is required with --wandb-project")
    started = time.monotonic()
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    print("DEVICES", devices, flush=True)
    if not tiny:
        assert len(devices) == 1 and devices[0].platform == "gpu", devices
        assert "H200" in devices[0].device_kind, devices[0].device_kind
    if tiny:
        config = VlmConfig(
            SiglipVisionConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=4,
                patch_size=2,
                attn_backend=AttentionBackend.VANILLA,
                use_flash_attention=False,
            ),
            Qwen3Config(
                hidden_dim=8,
                intermediate_dim=16,
                num_layers=1,
                num_heads=2,
                num_kv_heads=2,
                head_dim=4,
                max_seq_len=32,
                use_sliding_window=False,
                attn_backend=AttentionBackend.VANILLA,
            ),
            16,
        )
        backend = Tokenizer(
            WordLevel(
                {
                    "[UNK]": 0,
                    "[BOS]": 1,
                    "[EOS]": 2,
                    "a": 3,
                    "red": 4,
                    "blue": 5,
                    "square": 6,
                    "on": 7,
                    "black": 8,
                    "background": 9,
                },
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="[BOS]", eos_token="[EOS]")
        processor = SiglipImageProcessor(size={"height": 4, "width": 4})
        source_config = config
    else:
        source_config = VlmConfig.from_checkpoints()
        cache = output.parent / "hf-cache"
        vision_path = cached_checkpoint(source_config.vision_checkpoint, cache)
        text_path = cached_checkpoint(source_config.text_checkpoint, cache)
        config = dataclasses.replace(source_config, vision_checkpoint=vision_path, text_checkpoint=text_path)
        tokenizer = AutoTokenizer.from_pretrained(text_path)
        processor = SiglipImageProcessor.from_pretrained(vision_path)
    trackers = (JsonFileTrackerConfig(output_path=str(output)),)
    if wandb_project:
        trackers += (
            WandbConfig(
                project=wandb_project,
                entity=wandb_entity or None,
                name=f"{run_prefix}-{output.name}",
                mode="online",
                resume="never",
                save_code=False,
                background=False,
            ),
        )
    trainer_config = TrainerConfig(
        id=output.name,
        num_train_steps=steps,
        train_batch_size=2,
        per_device_parallelism=2,
        seed=0,
        mp=jmp.get_policy("p=f32,c=bf16,o=f32"),
        tracker=trackers,
        watch=WatchConfig(include_per_parameter_norms=False),
        log_dir=output / "logs",
        log_jaxprs=False,
        log_xla_hlo=False,
        require_accelerator=not tiny,
        load_checkpoint=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        checkpointer=CheckpointerConfig(base_path=str(output / "checkpoints"), append_run_id_to_base_path=False),
    )
    trainer_config.initialize()
    losses = []
    with Trainer(trainer_config, optax.adam(1e-3), compute_loss) as trainer:
        if tiny:
            model = Vlm.init(config, key=jax.random.PRNGKey(0))
        else:
            model = load_pretrained(
                config,
                key=jax.random.PRNGKey(0),
                axis_mapping=trainer.parameter_axis_mapping,
            )
        model = inference_mode(model, False)
        mask = dataclasses.replace(model, vision=False, projector=True, language=False)
        state = trainer.initial_state(jax.random.PRNGKey(1), model=model, is_trainable=mask)
        model = None
        trainable_count = parameter_count(trainables_only(state.model, mask))
        assert trainable_count == parameter_count(state.model.projector)
        assert trainable_count < 10_000_000
        trainer.tracker.log_hyperparameters(
            {
                "vision_checkpoint": source_config.vision_checkpoint,
                "text_checkpoint": source_config.text_checkpoint,
                "trainable_parameters": trainable_count,
                "training_stage": "projector",
                "learning_rate": 1e-3,
                "batch_size": 2,
                "text_length": 16 if tiny else 32,
                "steps": steps,
                "data": "two generated colored-square image/caption pairs, repeated",
                "tiny": tiny,
            }
        )
        print("TRAINABLE_PROJECTOR_PARAMETERS", trainable_count, flush=True)
        before = {name: checksum(getattr(state.model, name)) for name in ("vision", "language", "projector")}
        records = []
        for color in ("red", "blue"):
            image = Image.new("RGB", (config.vision.image_size, config.vision.image_size), "black")
            draw = ImageDraw.Draw(image)
            size = config.vision.image_size
            draw.rectangle((size // 4, size // 4, 3 * size // 4, 3 * size // 4), fill=color)
            records.append({"image": image, "text": f"a {color} square on a black background"})
        batch = caption_batch(records, tokenizer, processor, text_length=16 if tiny else 32)
        batch = hax.shard(batch, trainer.compute_axis_mapping)

        def record_step(info):
            if int(info.step) < len(losses):
                return
            loss = float(info.loss)
            assert math.isfinite(loss), loss
            losses.append(loss)
            print(json.dumps({"step": int(info.step), "loss": loss, "duration": float(info.step_duration)}), flush=True)

        trainer.add_hook(record_step, every=1)
        info = trainer.train(state, iter([batch] * steps))
        after = {name: checksum(getattr(info.state.model, name)) for name in ("vision", "language", "projector")}
        assert before["vision"] == after["vision"], "Vision weights changed"
        assert before["language"] == after["language"], "Language weights changed"
        assert before["projector"] != after["projector"], "Projector did not update"
        assert losses[-1] < losses[0], losses
        eqx.tree_serialise_leaves(output / "projector.eqx", info.state.model.projector)
    summary = {
        "status": "passed",
        "device": devices[0].device_kind,
        "steps": steps,
        "data": "two generated colored-square image/caption pairs, repeated",
        "trainable_parameters": trainable_count,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": losses,
        "frozen_vision_unchanged": before["vision"] == after["vision"],
        "frozen_language_unchanged": before["language"] == after["language"],
        "projector_changed": before["projector"] != after["projector"],
        "checksums_before": before,
        "checksums_after": after,
        "elapsed_seconds": time.monotonic() - started,
        "vision_checkpoint": source_config.vision_checkpoint,
        "text_checkpoint": source_config.text_checkpoint,
    }
    if wandb_project:
        assert wandb.run is not None
        summary["wandb_run_url"] = wandb.run.url
    trainer.tracker.log_summary(summary)
    trainer.tracker.finish()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PROJECTOR_SMOKE_RESULT", json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", default="/tmp/marin-projector-local-smoke")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--run-prefix", default="")
    args = parser.parse_args()
    run(args.output, args.steps, args.tiny, args.wandb_project, args.wandb_entity, args.run_prefix)
