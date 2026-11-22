import os
import signal
import subprocess

import modal

NEWTON = "newton[examples] @ git+https://github.com/skild-ai/newton.git@f6fc82a8b06d036e1e6d02a61d77a2d300476e8d"
SCENES = ["robot_panda_hydro", "franka_plate_rack", "box_turn", "drawer_fork", "toaster", "robot_ur10", "robot_allegro_hand"]
EXAMPLES = "/steel/examples"
TOOLS = f"{EXAMPLES}/scripts/render_audit"
RUN = "/out/run"
EGL_VENDOR = '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}'

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libegl1", "libgl1", "libglvnd0")
    .run_commands(f"mkdir -p /usr/share/glvnd/egl_vendor.d && echo '{EGL_VENDOR}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    .uv_pip_install("tomlkit>=0.13", NEWTON)
    .env({
        "PYTHON": "python",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "WARP_CACHE_PATH": "/out/cache/warp",
        "NEWTON_CACHE_PATH": "/out/cache/newton",
    })
    .add_local_dir("steel/examples", EXAMPLES, ignore=["docs", "**/__pycache__"])
)
app = modal.App("steel-render", image=image)
volume = modal.Volume.from_name("steel-render", create_if_missing=True)


@app.function(timeout=600, volumes={"/out": volume})
def plan():
    subprocess.run(["rm", "-rf", RUN])
    subprocess.run(["python", f"{TOOLS}/plan.py", EXAMPLES, f"{RUN}/plan.json", *SCENES], check=True)
    volume.commit()


@app.function(gpu="L4", timeout=3600, volumes={"/out": volume})
def render(scene):
    volume.reload()
    logs = subprocess.Popen(f"tail -qF {RUN}/out/{scene}/render.log {RUN}/out/{scene}/check.log 2>/dev/null | sed -u 's/^/[{scene}] /'", shell=True, start_new_session=True)
    subprocess.run(["bash", f"{TOOLS}/render_one.sh", RUN, scene])
    os.killpg(logs.pid, signal.SIGTERM)
    volume.commit()


@app.function(timeout=1800, volumes={"/out": volume})
def collect():
    volume.reload()
    subprocess.run(["python", f"{TOOLS}/collect.py", RUN], check=True)
    subprocess.run(["python", f"{TOOLS}/build_gallery.py", RUN], check=True)
    volume.commit()


@app.function(volumes={"/out": volume}, scaledown_window=60)
@modal.web_server(8000)
def gallery():
    subprocess.Popen(["python", "-m", "http.server", "8000", "--directory", RUN])


@app.local_entrypoint()
def main():
    plan.remote()
    list(render.map(SCENES))
    collect.remote()
    for path in ["index.html", "results.json", "media"]:
        subprocess.run(["modal", "volume", "get", "--force", "steel-render", f"run/{path}", "renders/"], check=True)
    print(f"local gallery: renders/index.html\nweb gallery: {modal.Function.from_name('steel-render', 'gallery').get_web_url()}")
