"""Hugging Face Space entry point (Gradio SDK, free tier).

Downloads Indexa from GitHub at startup and serves the FastAPI app on port 7860, so a
restart of the Space always runs the latest commit on main.
"""

import io
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import spaces  # ZeroGPU hardware expects at least one @spaces.GPU function

REPO = "RayidAlshammari/indexa"
DEST = Path("/tmp/indexa-src")


def fetch_service() -> Path:
    shutil.rmtree(DEST, ignore_errors=True)
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/main"
    with urllib.request.urlopen(url, timeout=180) as resp:
        tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz").extractall(DEST, filter="data")
    return next(DEST.iterdir()) / "services" / "ai-service"


@spaces.GPU
def _gpu_unused():
    return None


service = fetch_service()
sys.path.insert(0, str(service / "src"))
os.environ.setdefault("INDEXA_DATA_DIR", "/tmp/indexa-data")
# Gradio 6 otherwise starts a Node SSR server that takes port 7860 before uvicorn binds it.
os.environ["GRADIO_SSR_MODE"] = "false"

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from indexa.server import app  # noqa: E402

with gr.Blocks() as status:
    gr.Markdown("Indexa API is running. The demo page is at `/`.")
app = gr.mount_gradio_app(app, status, path="/gradio", ssr_mode=False)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
