# Baseline run: Inkling 2 Mini as a robot-use agent on LIBERO

## Goal

Measure Inkling 2 Mini's zero-shot success as a robot-use agent under PEAR's exact protocol, before any training. This answers three questions:

1. Does the pipeline work at all? (a frontier model must pass it)
2. Where does Inkling 2 Mini start, and how does that compare with PEAR's base model (Qwen3.8-27B: 1.6% on LIBERO)?
3. Which tasks give any success? Those are where RL training starts.

## Benchmark: LIBERO, with PEAR's harness unchanged

Use LIBERO for this run, not Steel:

- **There's a reference number.** PEAR reports Qwen3.8-27B base at ~2% on the goal/spatial/object suites and 0.4% on LIBERO-Long, and frontier models at 48–81%, all with the same protocol. A Steel scene has no reference yet, so a low number there could mean a bad model or a bad harness.
- **The code exists.** `pear/envs/libero/` (env + motion controller), `pear/envs/system_prompt.md`, `pear/agent_loop/protocol.py` and `tool_parser.py`.
- **Simulation runs on CPU** (MuJoCo + Mesa software rendering), so it's cheap on Modal.
- **It's PEAR's stage-1 training set,** so later RL results stay comparable.

Keep the harness identical to PEAR's for this run. The harness improvements from the literature (depth, back-projection, robot geometry in the prompt, motion cutoffs) are a separate, second run. Changing both at once makes the number uninterpretable.

## Episodes

| Set | Suites | Tasks | Trials | Samples per trial | Episodes |
|---|---|---|---|---|---|
| Control (frontier model) | libero_goal | 0-9 | 0-2 | 1 | 30 |
| Inkling main | libero_goal, libero_spatial, libero_object | 0-9 each | 0-1 | 4 | 240 |
| Inkling long | libero_10 | 0-9 | 0 | 2 | 20 |

4 samples per trial gives a per-task pass rate and tells you whether a group of 4 would contain any success, which is what GRPO needs.

Episode args are `{"suite": ..., "task_id": ..., "trial_id": ...}` (see `pear/envs/libero/configs/libero_gso_all.json` for the format).

## Setup

**Simulator (Modal, CPU).**
- Image: LIBERO at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` with Python 3.10 (pins in `pear/scripts/install/versions.env`), robosuite, mujoco, and Mesa for software EGL rendering.
- `LiberoEnv.setup_env` shows the environment variables it needs: `MUJOCO_GL=egl`, `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`, the Mesa EGL vendor file, and `CUDA_VISIBLE_DEVICES=""`.
- `LiberoEnv(suite, task_id, trial_id).rollout(history, action)` is stateless: it replays `history` (the low-level actions so far) and executes one protocol action. Call it directly from the agent loop. PEAR's Ray server, verl and the trainer aren't needed.

**Model.**
- Inkling 2 Mini served with vLLM or SGLang on a Modal B200 (native NVFP4) behind an OpenAI-compatible endpoint, or Tinker's sampling API.
- Confirm before running:
  - multi-image chat input works
  - how reasoning is returned: a separate `reasoning_content` field, or a think-end marker in the text, which PEAR's parser strips via `THINK_END`
  - which effort setting is used (record it)

**Agent loop** (~100 lines, one Modal script), following `pear/agent_loop/agent_loop.py`:

1. `result = env.rollout([], None)`. The system prompt is `system_prompt.md` with LIBERO's `environment_prompt.md` and `result["instruction"]` filled in.
2. The user turn is `json.dumps(result["state"])` plus the reminder sentence from `pear/envs/tool.py`, then `[agentview]` + image and `[wrist]` + image.
3. Generate, then take the last ```json block after the thinking. Validate it with `JsonBlockProtocol.check_action` against `MotionController.SCHEMA`.
4. `result = env.rollout(history, action)` and append `result["actions"]` to `history`.
5. Stop on `result["done"]`, a parse failure, or the token budget. Success means `done` and a positive reward.

**Limits:**
- Suite step limits are enforced by `LiberoEnv` (220-520 low-level steps).
- Episode budget: 128k response tokens.
- Per move: 32k thinking tokens (PEAR's real-robot limit).

## Order of runs

1. **Control run, ~$10.** Claude Sonnet 5.5 on the 30 control episodes. Any clear success rate (roughly 20%+) means the env, prompt and parser work. Near zero means debug the harness first: read the turn logs, check image orientation (PEAR rotates frames 180° and swaps left/right in the instruction), and check action parsing.
2. **Inkling main + long.** Only after the control run passes.
3. **Optional: Qwen3.8-27B on 30 episodes.** This should land near PEAR's ~2%, which confirms our reimplementation matches theirs.

## What to record per episode

- suite, task, trial, sample, success, end reason (`env` / `parse` / `budget`)
- moves, low-level steps
- thinking tokens per move, total input / output / cached tokens
- wall time, cost
- every turn's full text, as jsonl (needed for reading failures)

## Report

- Success by suite and overall, next to PEAR's numbers for Qwen3.8-27B base (~2% / 0.4%) and trained (74.4%).
- Per-task pass rate, and the share of tasks with at least one success in 4 samples.
- End-reason breakdown. A high `parse` share means a formatting problem, not a robotics problem; fix it before reading success numbers.
- Tokens per move, moves per episode, cost per episode and **cost per success**.
- 3-5 read-through failure cases, sorted into: wrong target, wrong pose or orientation, gripper timing, or never finished.

## Rough budget

- Control: ~$10 of Claude.
- Inkling: 260 episodes × ~50k tokens ≈ 13M tokens, a few B200-hours ≈ $15-40, or ~$10-20 on Tinker sampling.
- Simulator CPU: a few dollars.

## After this

- **If Inkling gets any success on some tasks:** run the harness-improvement variant on the same episodes, then port the protocol to a Steel tabletop scene.
- **If it gets zero everywhere:** look at the end reasons first. Then try the harness improvements, which the literature shows can move success several-fold, before considering a fine-tuning warm start.
