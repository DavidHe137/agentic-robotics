import modal
import newton
import torch
import warp as wp
from newton.solvers import SolverSteel
from torch import nn

NEWTON = "newton[examples] @ git+https://github.com/skild-ai/newton.git@f6fc82a8b06d036e1e6d02a61d77a2d300476e8d"

image = modal.Image.debian_slim(python_version="3.12").apt_install("git").uv_pip_install("torch", "viser>=1.1.1", NEWTON)
app = modal.App("steel-cartpole-ppo", image=image)
volume = modal.Volume.from_name("steel-cartpole", create_if_missing=True)

WORLDS = 4096
HORIZON = 32
ITERATIONS = 200
MAX_STEPS = 500
FRAME_DT = 1.0 / 60.0
SUBSTEPS = 2


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def cartpole(worlds, device):
    world = newton.ModelBuilder()
    cart = world.add_link(xform=tf(0.0, 0.0, 1.0))
    world.add_shape_box(cart, hx=0.2, hy=0.1, hz=0.05)
    pole = world.add_link(xform=tf(0.0, 0.0, 1.5))
    world.add_shape_capsule(pole, radius=0.02, half_height=0.5)
    slider = world.add_joint_prismatic(-1, cart, axis=(1.0, 0.0, 0.0), parent_xform=tf(0.0, 0.0, 1.0))
    hinge = world.add_joint_revolute(cart, pole, axis=(0.0, 1.0, 0.0), child_xform=tf(0.0, 0.0, -0.5))
    world.add_articulation([slider, hinge])

    builder = newton.ModelBuilder()
    builder.replicate(world, worlds, spacing=(0.0, 0.0, 0.0))
    return builder.finalize(device=device)


def mlp():
    return nn.Sequential(nn.Linear(4, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))


@app.function(gpu="L4", timeout=3600, volumes={"/out": volume})
def train():
    model = cartpole(WORLDS, "cuda")
    solver = SolverSteel(model)
    state_0, state_1, control = model.state(), model.state(), model.control()
    world_mask = wp.zeros(WORLDS + 1, dtype=wp.bool)

    def reset(done):
        q = wp.to_torch(state_0.joint_q).view(WORLDS, 2)
        qd = wp.to_torch(state_0.joint_qd).view(WORLDS, 2)
        q[done] = (torch.rand(int(done.sum()), 2, device="cuda") - 0.5) * 0.5
        qd[done] = (torch.rand(int(done.sum()), 2, device="cuda") - 0.5) * 0.5
        wp.to_torch(world_mask)[:WORLDS] = done
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0, mask=world_mask[:WORLDS])
        solver.reset(state_0, world_mask=world_mask)

    def observe():
        q = wp.to_torch(state_0.joint_q).view(WORLDS, 2)
        qd = wp.to_torch(state_0.joint_qd).view(WORLDS, 2)
        return torch.cat([q, qd], dim=1)

    def step(action):
        nonlocal state_0, state_1
        wp.to_torch(control.joint_f).view(WORLDS, 2)[:, 0] = 20.0 * action.clamp(-1.0, 1.0)
        for _ in range(SUBSTEPS):
            solver.step(state_0, state_1, control, None, FRAME_DT / SUBSTEPS)
            state_0, state_1 = state_1, state_0

    actor, critic = mlp().cuda(), mlp().cuda()
    log_std = nn.Parameter(torch.zeros(1, device="cuda"))
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters(), log_std], lr=3e-4)

    reset(torch.ones(WORLDS, dtype=torch.bool, device="cuda"))
    steps = torch.zeros(WORLDS, device="cuda")
    for it in range(ITERATIONS):
        obs, actions, logps, rewards, dones, values, lengths = [], [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(HORIZON):
                o = observe()
                dist = torch.distributions.Normal(actor(o).squeeze(-1), log_std.exp())
                a = dist.sample()
                step(a)
                x, theta = observe()[:, :2].T
                steps += 1
                done = (x.abs() > 2.4) | (theta.abs() > 0.4) | (steps >= MAX_STEPS)
                obs.append(o), actions.append(a), logps.append(dist.log_prob(a)), values.append(critic(o).squeeze(-1))
                rewards.append(1.0 - 0.1 * x**2), dones.append(done.float())
                lengths.append(steps[done])
                steps[done] = 0
                reset(done)
            next_value = critic(observe()).squeeze(-1)

            advantages = torch.zeros(HORIZON, WORLDS, device="cuda")
            gae = torch.zeros(WORLDS, device="cuda")
            for t in reversed(range(HORIZON)):
                next_v = next_value if t == HORIZON - 1 else values[t + 1]
                delta = rewards[t] + 0.99 * next_v * (1 - dones[t]) - values[t]
                gae = delta + 0.99 * 0.95 * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + torch.stack(values)

        obs, actions, logps = torch.cat(obs), torch.cat(actions), torch.cat(logps)
        advantages, returns = advantages.flatten(), returns.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(4):
            for idx in torch.randperm(len(obs), device="cuda").chunk(4):
                dist = torch.distributions.Normal(actor(obs[idx]).squeeze(-1), log_std.exp())
                ratio = (dist.log_prob(actions[idx]) - logps[idx]).exp()
                policy_loss = -torch.min(ratio * advantages[idx], ratio.clamp(0.8, 1.2) * advantages[idx]).mean()
                value_loss = (critic(obs[idx]).squeeze(-1) - returns[idx]).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss - 0.005 * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
                optimizer.step()

        finished = torch.cat(lengths)
        print(f"iter {it:3d}  episodes {len(finished):5d}  mean length {finished.mean().item():6.1f}  std {log_std.exp().item():.2f}")

    torch.save(actor.cpu().state_dict(), "/out/policy.pt")


@app.function(timeout=3600, volumes={"/out": volume})
def view():
    import time

    model = cartpole(1, "cpu")
    solver = SolverSteel(model)
    state_0, state_1, control = model.state(), model.state(), model.control()
    actor = mlp()
    actor.load_state_dict(torch.load("/out/policy.pt"))

    def reset():
        wp.to_torch(state_0.joint_q)[:] = torch.tensor([0.0, 0.2])
        wp.to_torch(state_0.joint_qd)[:] = 0.0
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        solver.reset(state_0)

    viewer = newton.viewer.ViewerViser(port=8080)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(0.0, -7.0, 1.5), pitch=0.0, yaw=90.0)
    viewer.set_reset_callback(reset)
    reset()

    with modal.forward(8080) as tunnel:
        print(f"open {tunnel.url}")
        t = 0.0
        while viewer.is_running():
            if not viewer.is_paused():
                obs = torch.cat([wp.to_torch(state_0.joint_q), wp.to_torch(state_0.joint_qd)])
                with torch.no_grad():
                    wp.to_torch(control.joint_f)[0] = 20.0 * actor(obs).clamp(-1.0, 1.0)
                for _ in range(SUBSTEPS):
                    state_0.clear_forces()
                    viewer.apply_forces(state_0)
                    solver.step(state_0, state_1, control, None, FRAME_DT / SUBSTEPS)
                    state_0, state_1 = state_1, state_0
                if not (wp.to_torch(state_0.joint_q).abs() < torch.tensor([2.4, 1.0])).all():
                    reset()
                t += FRAME_DT
                if round(t / FRAME_DT) % 60 == 0:
                    print(f"t {t:6.1f}  x {state_0.joint_q.numpy()[0]:6.2f}  theta {state_0.joint_q.numpy()[1]:6.2f}")
            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.end_frame()
            time.sleep(FRAME_DT)
