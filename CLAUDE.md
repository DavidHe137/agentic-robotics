# Inkling

Robot learning experiments on Steel (Skild's GPU rigid-body solver inside Newton), run on Modal. Everything is sim-only for now.

## Direction

- RL environments on Steel with thousands of parallel worlds (`train.py` is the cartpole PPO baseline).
- Agentic sim workflows: agents build and tune scenes, train or script experts, grade them, and fan out data generation as parallel Modal jobs.
- Turn Thinking Machines' Inkling 2 Mini (open-weight looped MoE VLM, 26.4B total / 3.4B active, https://thinkingmachines.ai/news/inkling-2-mini/) into a robot-use agent, following PEAR (https://arxiv.org/abs/2611.18562): each round the model sees camera images and end-effector state as JSON, reasons, and outputs an end-effector pose delta as JSON; IK and a controller execute it. Trained with RL only (GRPO, binary success reward), no demonstrations, on a curriculum of tasks.
- Steel is the task factory: agents build scenes in Steel, each with an instruction and a success check, rendered with `newton.sensors.SensorTiledCamera`.
- Planned order: the robot-use protocol on one Franka tabletop scene in Steel, base-model success rates across a handful of tasks, then RL, then agent-generated tasks.

## Running

- The local machine is a Mac with no CUDA. All GPU work runs on Modal: `uv run modal run <script>.py::<function>`.
- Newton is pinned to a commit of the `skild-ai/newton` fork in two places: `pyproject.toml` and the image definition in each Modal script. Keep them identical.
- Use uv for everything Python.

## steel/

`steel/` is an untracked, read-only copy of the upstream Steel repo. Do not edit it. Start from `steel/agents/README.md` for the workflows and `steel/examples/AGENTS.md` for examples and solver presets. Its render-and-grade tooling assumes local GPUs with EGL and ffmpeg, so running it on Modal needs a custom image (`render.py` runs render-and-grade on Modal and serves the gallery).
- Steel also has a WebGPU implementation that runs in the browser (https://github.com/skild-ai/steel-webgpu, demos at https://www.skild.ai/steel/playground). Where a scene already exists there (the shelf's ported scenes, e.g. the hero-montage tabletop demos like `toaster` and `jenga`, see `steel/examples/README.md` "Ported scenes"), it is worth trying in the browser as a lighter alternative to rendering on Modal. Use Chrome, not Safari.

## Code style

- One self-contained Modal script per experiment, constants at the top.
- No comments, docstrings, or READMEs.
- Simple code: avoid new types and helper functions unless necessary, no defensive None checks.

## Working alongside other agents

Several agents work in this repo at the same time. Do not edit, revert, or commit files you did not create or were not asked to change. Stage files by name, never `git add -A`.
