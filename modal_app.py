"""Run the pinned Marin projector smoke test on one Modal H200."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REPOSITORY_URL = "https://github.com/bkolligs/marin-modal.git"
REMOTE_ROOT = "/workspace/marin-modal"
REMOTE_MARIN = f"{REMOTE_ROOT}/marin"
PYTHON = "/opt/marin-venv/bin/python"
RESULT_VOLUME = "marin-vlm-projector-smoke-results"


def repository_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT if modal.is_local() else Path(REMOTE_ROOT), text=True
    ).strip()


app = modal.App("marin-vlm-projector-smoke")
results_volume = modal.Volume.from_name(RESULT_VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("uv==0.10.11")
    .run_commands(
        f"git clone --filter=blob:none --no-checkout {REPOSITORY_URL} {REMOTE_ROOT}",
        f"git -C {REMOTE_ROOT} fetch --depth 1 origin {repository_revision()}",
        f"git -C {REMOTE_ROOT} checkout --detach {repository_revision()}",
        f"git -C {REMOTE_ROOT} submodule update --init --depth 1 marin",
        f"cd {REMOTE_MARIN} && UV_PROJECT_ENVIRONMENT=/opt/marin-venv "
        "uv sync --python 3.12 --frozen --package marin-levanter --no-default-groups",
        f"uv pip install --python {PYTHON} 'jax[cuda13]==0.11.1' 'pillow==13.0.0'",
        f"PYTHONPATH={REMOTE_MARIN} {PYTHON} -c 'import levanter.trainer; import experiments.multimodality.base.train'",
    )
    .env(
        {
            "PYTHONPATH": REMOTE_MARIN,
            "HF_HOME": "/tmp/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "JAX_PLATFORMS": "cuda,cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
)


@app.function(
    image=image,
    gpu="H200",
    cpu=8,
    memory=65536,
    timeout=1200,
    retries=0,
    max_containers=1,
    volumes={"/results": results_volume},
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
)
def projector_smoke(steps: int = 20, wandb_project: str = "marin-vlm", wandb_entity: str = "") -> dict:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], check=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path("/results") / run_id
    try:
        subprocess.run(
            [
                PYTHON,
                "-u",
                f"{REMOTE_ROOT}/projector_workload.py",
                "--steps",
                str(steps),
                "--output",
                str(output),
                "--wandb-project",
                wandb_project,
                "--wandb-entity",
                wandb_entity,
            ],
            cwd=REMOTE_MARIN,
            check=True,
        )
        summary = json.loads((output / "summary.json").read_text())
        summary["modal_result_path"] = str(output)
        summary["source_commit"] = subprocess.check_output(
            ["git", "-C", REMOTE_MARIN, "rev-parse", "HEAD"], text=True
        ).strip()
        summary["launcher_commit"] = subprocess.check_output(
            ["git", "-C", REMOTE_ROOT, "rev-parse", "HEAD"], text=True
        ).strip()
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        results_volume.commit()


@app.function(image=image, timeout=300)
def environment_info() -> dict:
    """Inspect the initialized remote checkout without requesting a GPU."""
    return {
        "launcher_commit": subprocess.check_output(["git", "-C", REMOTE_ROOT, "rev-parse", "HEAD"], text=True).strip(),
        "marin_commit": subprocess.check_output(["git", "-C", REMOTE_MARIN, "rev-parse", "HEAD"], text=True).strip(),
        "submodule_status": subprocess.check_output(
            ["git", "-C", REMOTE_ROOT, "submodule", "status"], text=True
        ).strip(),
        "python": subprocess.check_output([PYTHON, "--version"], text=True).strip(),
        "workspace": REMOTE_ROOT,
    }


@app.local_entrypoint()
def main(steps: int = 20, wandb_project: str = "marin-vlm", wandb_entity: str = "") -> None:
    result = projector_smoke.remote(steps, wandb_project, wandb_entity)
    output = ROOT / "run-results"
    output.mkdir(exist_ok=True)
    (output / "latest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
