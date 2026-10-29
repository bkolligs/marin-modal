# marin-modal

Modal launch tooling for the Marin VLM experiment. The `marin/` Git submodule
pins the experiment code and its workspace dependency lock. Modal builds clone
this repository at the launcher's commit and initialize the recorded submodule
commit inside the container.

## Initialize

```bash
git clone --recurse-submodules https://github.com/bkolligs/marin-modal.git
cd marin-modal
uv sync --python 3.12
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

Authenticate the Modal CLI once with `uv run modal token new` if no profile is
already configured. Credentials stay outside this repository.

Inspect the initialized Modal checkout without requesting a GPU:

```bash
uv run modal run modal_app.py::environment_info
```

## H200 projector smoke test

Commit and push launcher changes before running: the image clones the launcher's
committed revision from GitHub. Local uncommitted edits are not copied into the
remote checkout.

```bash
uv run modal run modal_app.py --steps 20
```

The function requests one H200, eight CPU cores, and 64 GiB of host memory, with a
20-minute execution timeout and no retries. It loads pinned SigLIP and Delphi
998M weights, repeats two generated colored-square image/caption pairs, and trains
only the projector using Levanter's optimizer parameter mask. It checks finite,
decreasing loss, changed projector weights, and unchanged backbone checksums.
This is an execution smoke test, not a dataset-quality evaluation.

Results, projector checkpoints, and the pinned backbone download cache are saved in the
`marin-vlm-projector-smoke-results` Modal Volume. The local result is written to
`run-results/latest.json`, which is ignored by Git.

## Interactive environment

Open a shell with the same image and resources:

```bash
uv run modal shell modal_app.py::projector_smoke
```

Inside that container, the launcher checkout is `/workspace/marin-modal`, the
initialized submodule is `/workspace/marin-modal/marin`, and the training Python
interpreter is `/opt/marin-venv/bin/python`. Use `/results` for persistent files.
An H200 shell incurs compute charges while it remains running.

## Updating Marin

```bash
git -C marin fetch origin <tested-commit>
git -C marin checkout --detach FETCH_HEAD
git add marin
git commit -m "chore: update the Marin submodule"
git push
```

The recorded submodule commit is the run's source of truth. Moving the fork's
branch alone does not change the code used by an existing launcher commit.
