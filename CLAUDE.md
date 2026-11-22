# Inkling

Robot learning experiments on Steel (Skild's GPU rigid-body solver inside Newton), run on Modal. Everything is sim-only for now.

## Direction

- RL environments on Steel with thousands of parallel worlds (`train.py` is the cartpole PPO baseline).
- Agentic sim workflows: agents build and tune scenes, train or script experts, grade them, and fan out data generation as parallel Modal jobs.
- Turn Thinking Machines' Inkling 2 Mini (open-weight looped MoE VLM, 26.4B total / 3.4B active, https://thinkingmachines.ai/news/inkling-2-mini/) into a robot-use agent, following PEAR (https://arxiv.org/abs/2611.18562): each round the model sees camera images and end-effector state as JSON, reasons, and outputs an end-effector pose delta as JSON; IK and a controller execute it. Trained with RL only (GRPO, binary success reward), no demonstrations, on a curriculum of tasks.
- Steel is the task factory: agents build scenes in Steel, each with an instruction and a success check, rendered with `newton.sensors.SensorTiledCamera`.
- Planned order: the robot-use protocol on one Franka tabletop scene in Steel, base-model success rates across a handful of tasks, then RL, then agent-generated tasks.

## Running

- The local machine is a Mac with no CUDA. All GPU work runs on Modal: `uv run modal run <script>.py::<function>`. Run modal as its own command (no `cd ... &&` before it, no pipe after it): only commands that start with `modal` or `uv run modal` are exempt from the sandbox, and others fail with "Could not connect to the Modal server".
- Newton is pinned to a commit of the `skild-ai/newton` fork in two places: `pyproject.toml` and the image definition in each Modal script. Keep them identical.
- Use uv for everything Python. Run `uv` from the repo root, never from inside `steel/` or `pear/` (it would create a venv for their projects).
- Credentials live only in Modal secrets; never put keys in code, files, or chat. Attach them with `secrets=[modal.Secret.from_name(...)]`:
  - `anthropic`: `ANTHROPIC_API_KEY` (verified working). For Claude as a frontier baseline or teacher.
  - `tinker`: `TINKER_API_KEY` (verified working, SDK `tinker` 0.39.2). For Inkling 2 Mini sampling and training on Tinker. Model names: `thinkingmachines/Inkling-2-Mini` (64K context) and `thinkingmachines/Inkling-2-Mini:peft:262144` (256K). List models with `tinker.RestClient().get_server_capabilities().result().supported_models`.
  - `cloudflare_r2`: pre-existing, unrelated to this project's plan so far.
- The first experiment is the LIBERO baseline in `baseline.md`: Inkling 2 Mini zero-shot under PEAR's unchanged harness, after a Claude control run.

## pear/

`pear/` is an untracked, read-only clone of https://github.com/mit-han-lab/pear (the PEAR paper's code). Reuse its protocol and LIBERO env directly: `pear/envs/system_prompt.md`, `pear/envs/libero/` (env, motion controller, environment prompt), `pear/agent_loop/protocol.py` and `tool_parser.py`. Its Ray/verl trainer and Qwen kernels are not needed.

## steel/

`steel/` is an untracked, read-only copy of the upstream Steel repo. Do not edit it. Start from `steel/agents/README.md` for the workflows and `steel/examples/AGENTS.md` for examples and solver presets. Its render-and-grade tooling assumes local GPUs with EGL and ffmpeg, so running it on Modal needs a custom image (`render.py` runs render-and-grade on Modal and serves the gallery).
- Steel also has a WebGPU implementation that runs in the browser (https://github.com/skild-ai/steel-webgpu, demos at https://www.skild.ai/steel/playground). Where a scene already exists there (the shelf's ported scenes, e.g. the hero-montage tabletop demos like `toaster` and `jenga`, see `steel/examples/README.md` "Ported scenes"), it is worth trying in the browser as a lighter alternative to rendering on Modal. Use Chrome, not Safari.

## Code style

- One self-contained Modal script per experiment, constants at the top.
- No comments, docstrings, or READMEs.
- Simple code: avoid new types and helper functions unless necessary, no defensive None checks.

## Working alongside other agents

Several agents work in this repo at the same time. Do not edit, revert, or commit files you did not create or were not asked to change. Stage files by name, never `git add -A`.
