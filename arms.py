from pathlib import Path

import modal
import newton
import newton.ik as ik
import numpy as np
import warp as wp
from newton.sensors import SensorCamera
from newton.solvers import SolverSteel

NEWTON = "newton[examples] @ git+https://github.com/skild-ai/newton.git@f6fc82a8b06d036e1e6d02a61d77a2d300476e8d"
SO_ARM100 = ("https://github.com/TheRobotStudio/SO-ARM100.git", "141aea7bcac2744d92998e30268c413202a56f96", "Simulation/SO101")
I2RT = ("https://github.com/i2rt-robotics/i2rt.git", "8c1b3ab6b6f31ae11e8b5f1e46c037d154199938", "i2rt/robot_models/arm/yam")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("coacd", "imageio[ffmpeg]", NEWTON)
    .run_commands(*[
        f"git clone --filter=blob:none --sparse {url} /models/{Path(url).stem} && cd /models/{Path(url).stem} && git sparse-checkout set {path} && git checkout {sha}"
        for url, sha, path in [SO_ARM100, I2RT]
    ])
    .env({"WARP_CACHE_PATH": "/out/cache/warp"})
)
app = modal.App("steel-arms", image=image)
volume = modal.Volume.from_name("steel-arms", create_if_missing=True)

ARMS = {
    "so101": dict(
        urdf="/models/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        visuals_as_colliders=False,
        decompose=["gripper_link", "moving_jaw_so101_v1_link"],
        ee="gripper_link",
        tcp=(0.375, 0.0, 0.238),
        fingers=1,
        grip=(1.2, -0.1),
        grip_gains=(20.0, 1.0, 3.0),
        cube_x=0.22,
    ),
    "yam": dict(
        urdf="/models/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
        visuals_as_colliders=True,
        decompose=["tip_left", "tip_right"],
        ee="gripper",
        tcp=(0.235, 0.0, 0.1735),
        fingers=2,
        grip=(-0.047, 0.0),
        grip_gains=(1000.0, 18.0, 30.0),
        cube_x=0.32,
    ),
}

CUBE = 0.015
FPS = 60
SUBSTEPS = 8
ITERATIONS = 12
WIDTH, HEIGHT = 640, 480
KEYS = [(0.0, 0.12, 0), (1.0, 0.12, 0), (2.0, 0.0, 0), (2.8, 0.0, 1), (4.3, 0.15, 1), (5.5, 0.15, 1)]


def load_arm(builder, arm):
    builder.add_urdf(arm["urdf"], floating=False, enable_self_collisions=False, parse_visuals_as_colliders=arm["visuals_as_colliders"], joint_damping_mode="drive")


@app.function(gpu="L4", timeout=3600, volumes={"/out": volume})
def run(name):
    import imageio

    arm = ARMS[name]
    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = 0.75
    b.default_shape_cfg.gap = 0.001
    b.default_shape_cfg.density = 600
    load_arm(b, arm)
    coords = b.joint_coord_count
    for i in range(coords):
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_armature[i], b.joint_effort_limit[i] = 3000.0, 100.0, 0.08, 100.0
        b.joint_target_mode[i] = int(newton.JointTargetMode.POSITION)
    fingers = list(range(coords - arm["fingers"], coords))
    for i in fingers:
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = arm["grip_gains"]
    decompose = [i for i, body in enumerate(b.shape_body) if body >= 0 and b.body_label[body].split("/")[-1] in arm["decompose"]]
    if decompose:
        b.approximate_meshes("coacd", shape_indices=decompose, keep_visual_shapes=True, cache_dir="/out/cache/coacd")
    b.approximate_meshes("convex_hull", keep_visual_shapes=True)
    x = arm["cube_x"]
    cube = b.add_body(xform=wp.transform((x, 0.0, CUBE), wp.quat_identity()), label="cube")
    b.add_shape_box(cube, hx=CUBE, hy=CUBE, hz=CUBE, color=(0.85, 0.25, 0.2))
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.02), wp.quat_identity()), hx=0.6, hy=0.6, hz=0.02, color=(0.55, 0.6, 0.65))
    model = b.finalize()
    volume.commit()

    ik_builder = newton.ModelBuilder()
    load_arm(ik_builder, arm)
    ik_model = ik_builder.finalize()
    ee = next(i for i, label in enumerate(ik_model.body_label) if label.endswith("/" + arm["ee"]))
    home = ik_model.state()
    newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, home)
    ee_home = wp.transform(*home.body_q.numpy()[ee])
    tcp = wp.transform_point(wp.transform_inverse(ee_home), wp.vec3(*arm["tcp"]))
    down = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2) * wp.transform_get_rotation(ee_home)
    target_pos = wp.array([wp.vec3(x, 0.0, CUBE + KEYS[0][1])], dtype=wp.vec3)
    solver_ik = ik.IKSolver(
        model=ik_model,
        n_problems=1,
        objectives=[
            ik.IKObjectivePosition(link_index=ee, link_offset=tcp, target_positions=target_pos),
            ik.IKObjectiveRotation(link_index=ee, link_offset_rotation=wp.quat_identity(), target_rotations=wp.array([wp.vec4(*down)], dtype=wp.vec4)),
            ik.IKObjectiveJointLimit(joint_limit_lower=ik_model.joint_limit_lower, joint_limit_upper=ik_model.joint_limit_upper, weight=10.0),
        ],
        lambda_initial=0.05,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    q = wp.array(ik_model.joint_q.numpy()[None], dtype=float)
    q_np = q.numpy()
    q_np[0, fingers] = arm["grip"][0]
    q.assign(q_np)
    solver_ik.step(q, q, iterations=200)

    full_q = model.joint_q.numpy()
    full_q[:coords] = q.numpy()[0]
    model.joint_q.assign(full_q)
    model.joint_target_q.assign(np.concatenate([full_q[:coords], model.joint_target_q.numpy()[coords:]]))
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=16384)
    contacts = pipeline.contacts()
    solver = SolverSteel(model, solve_mode="serial", articulated_contact_response="immediate", iterations=ITERATIONS, enable_joint_limits=True)

    camera = SensorCamera(model)
    camera.create_default_light()
    rays = SensorCamera.compute_camera_rays_pinhole(WIDTH, HEIGHT, camera_fov=0.7, device=model.device)
    look = np.array([0.5 * x, 0.0, 0.08])
    eye = look + x * np.array([1.4, -2.0, 1.2])
    f = (look - eye) / np.linalg.norm(look - eye)
    r = np.cross(f, (0.0, 0.0, 1.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    rot = wp.quat_from_matrix(wp.mat33(*np.column_stack([r, u, -f]).flatten()))
    view = wp.array([wp.transform(wp.vec3(*eye), rot)], dtype=wp.transformf)
    color = camera.allocate_image_color(1, WIDTH, HEIGHT)

    times, heights, grips = np.array([k[0] for k in KEYS]), np.array([k[1] for k in KEYS]), np.array([k[2] for k in KEYS])
    frames, cube_z, err = [], [], []
    for frame in range(int(KEYS[-1][0] * FPS)):
        t = frame / FPS
        target_pos.assign(np.array([[x, 0.0, CUBE + np.interp(t, times, heights)]], dtype=np.float32))
        solver_ik.step(q, q, iterations=24)
        target = control.joint_target_q.numpy()
        target[:coords] = q.numpy()[0]
        target[fingers] = np.interp(np.interp(t, times, grips), (0, 1), arm["grip"])
        control.joint_target_q.assign(target)
        for _ in range(SUBSTEPS):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / FPS / SUBSTEPS)
            state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        tcp_now = wp.transform_point(wp.transform(*body_q[ee]), tcp)
        err.append(np.linalg.norm(np.array(tcp_now) - target_pos.numpy()[0]))
        cube_z.append(body_q[cube][2])
        if frame % 2 == 0:
            model.bvh_refit_shapes(state_0)
            camera.update(state_0, view, rays, color_image=color)
            frames.append(color.numpy()[0].view(np.uint8).reshape(HEIGHT, WIDTH, 4)[..., :3].copy())
        if frame % 30 == 0:
            print(f"[{name}] t {t:4.1f}  tcp err {err[-1] * 1000:6.1f} mm  cube z {cube_z[-1]:.3f}  q {np.round(state_0.joint_q.numpy()[:coords], 2)}")

    print(f"[{name}] initial ik err {err[0] * 1000:.1f} mm, max tracking err {max(err) * 1000:.1f} mm, final cube z {cube_z[-1]:.3f} (lifted: {cube_z[-1] > CUBE + 0.08})")
    path = f"/tmp/{name}.mp4"
    imageio.mimsave(path, frames, fps=FPS // 2)
    return Path(path).read_bytes()


@app.local_entrypoint()
def main():
    Path("renders/arms").mkdir(parents=True, exist_ok=True)
    for name, video in zip(ARMS, run.map(list(ARMS))):
        Path(f"renders/arms/{name}.mp4").write_bytes(video)
        print(f"renders/arms/{name}.mp4")
