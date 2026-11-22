# Inkling

Robot learning experiments on Steel (Skild's GPU rigid-body solver inside Newton), run on Modal. Everything is sim-only for now.

## Direction

- RL environments on Steel with thousands of parallel worlds (`train.py` is the cartpole PPO baseline).
- Agentic sim workflows: agents build and tune scenes, train or script experts, grade them, and fan out data generation as parallel Modal jobs.
- Inkling Mini: a small VLA adapted from SmolVLM. Data is rendered in sim with `newton.sensors.SensorTiledCamera` and labeled from task definitions. Checkpoints are evaluated closed-loop in the same Steel scenes.
- Planned order: a cartpole-level vision task end to end (render, dataset, train, eval), then SmolVLM with discrete action tokens on a few `steel/examples/shelf` tasks, then a flow-matching action expert and agent-driven task fan-out.

## Running

- The local machine is a Mac with no CUDA. All GPU work runs on Modal: `uv run modal run <script>.py::<function>`.
- Newton is pinned to a commit of the `skild-ai/newton` fork in two places: `pyproject.toml` and the image definition in each Modal script. Keep them identical.
- Use uv for everything Python.

## steel/

`steel/` is an untracked, read-only copy of the upstream Steel repo. Do not edit it. Start from `steel/agents/README.md` for the workflows and `steel/examples/AGENTS.md` for examples and solver presets. Its render-and-grade tooling assumes local GPUs with EGL and ffmpeg, so running it on Modal needs a custom image.

## Code style

- One self-contained Modal script per experiment, constants at the top.
- No comments, docstrings, or READMEs.
- Simple code: avoid new types and helper functions unless necessary, no defensive None checks.

## Working alongside other agents

Several agents work in this repo at the same time. Do not edit, revert, or commit files you did not create or were not asked to change. Stage files by name, never `git add -A`.
