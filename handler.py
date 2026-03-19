import base64
import io
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import requests
import runpod
from PIL import Image


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("vllm-omni-runpod")

SERVER_HOST = os.getenv("VLLM_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("VLLM_SERVER_PORT", "8091"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_URL = f"{SERVER_URL}/health"
MODELS_URL = f"{SERVER_URL}/v1/models"
IMAGES_URL = f"{SERVER_URL}/v1/images/generations"

STARTUP_TIMEOUT_SECONDS = int(os.getenv("VLLM_STARTUP_TIMEOUT", "3600"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("VLLM_REQUEST_TIMEOUT", "1800"))
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen-Image-2512")
DEFAULT_SERVER_ARGS = os.getenv("VLLM_SERVER_ARGS", "--num-gpus 1 --vae-use-slicing --vae-use-tiling")
DEFAULT_IMAGE_SIZE = os.getenv("DEFAULT_IMAGE_SIZE", "1328x1328")
DEFAULT_NUM_INFERENCE_STEPS = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", "50"))
DEFAULT_TRUE_CFG_SCALE = float(os.getenv("DEFAULT_TRUE_CFG_SCALE", "4.0"))
DEFAULT_GUIDANCE_SCALE = os.getenv("DEFAULT_GUIDANCE_SCALE")
DEFAULT_NEGATIVE_PROMPT = os.getenv("DEFAULT_NEGATIVE_PROMPT", "").strip()
LOG_TAIL_LINES = int(os.getenv("LOG_TAIL_LINES", "200"))

_SERVER_LOCK = threading.Lock()
_SERVER_PROCESS: subprocess.Popen[str] | None = None
_SERVER_MODEL: str | None = None
_SERVER_ARGS: tuple[str, ...] | None = None
_SERVER_LOG_TAIL: deque[str] = deque(maxlen=LOG_TAIL_LINES)


@dataclass(frozen=True)
class ResolvedServerConfig:
    model: str
    server_args: tuple[str, ...]


def _parse_server_args(value: Any, default: str | None = None) -> tuple[str, ...]:
    if value is None:
        return tuple(shlex.split(default or ""))
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise TypeError("server_args must be a string or list of strings")


def _resolve_server_config(job_input: dict[str, Any]) -> ResolvedServerConfig:
    model_value = job_input.get("model", DEFAULT_MODEL)
    server_args_value = job_input.get("server_args")
    model_name = DEFAULT_MODEL
    model_server_args = _parse_server_args(None, DEFAULT_SERVER_ARGS)

    if isinstance(model_value, str):
        model_name = model_value.strip() or DEFAULT_MODEL
    elif isinstance(model_value, dict):
        model_name = str(model_value.get("name") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        model_server_args = _parse_server_args(model_value.get("server_args"), DEFAULT_SERVER_ARGS)
    else:
        raise TypeError("model must be a string or object")

    if server_args_value is not None:
        model_server_args = _parse_server_args(server_args_value, DEFAULT_SERVER_ARGS)

    return ResolvedServerConfig(model=model_name, server_args=model_server_args)


def _coerce_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _coerce_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _parse_size(job_input: dict[str, Any]) -> tuple[int, int]:
    if "size" in job_input and job_input["size"]:
        raw = str(job_input["size"])
        parts = raw.lower().split("x", 1)
        if len(parts) != 2:
            raise ValueError("size must be in WIDTHxHEIGHT format")
        return _coerce_int(parts[0], name="size width"), _coerce_int(parts[1], name="size height")

    if "width" in job_input or "height" in job_input:
        if "width" not in job_input or "height" not in job_input:
            raise ValueError("width and height must be provided together")
        return _coerce_int(job_input["width"], name="width"), _coerce_int(job_input["height"], name="height")

    parts = DEFAULT_IMAGE_SIZE.lower().split("x", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid DEFAULT_IMAGE_SIZE: {DEFAULT_IMAGE_SIZE}")
    return int(parts[0]), int(parts[1])


def _reader_thread(stream: Any) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            clean = line.rstrip()
            _SERVER_LOG_TAIL.append(clean)
            LOGGER.info("[vllm] %s", clean)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _server_log_excerpt() -> str:
    if not _SERVER_LOG_TAIL:
        return "no server logs captured"
    return "\n".join(_SERVER_LOG_TAIL)


def _server_process_running() -> bool:
    return _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is None


def _server_healthy() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _stop_server_locked() -> None:
    global _SERVER_PROCESS, _SERVER_MODEL, _SERVER_ARGS

    if _SERVER_PROCESS is None:
        return

    process = _SERVER_PROCESS
    if process.poll() is None:
        LOGGER.info("Stopping vLLM-Omni server")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            process.terminate()

        deadline = time.time() + 60
        while time.time() < deadline and process.poll() is None:
            time.sleep(1)

        if process.poll() is None:
            LOGGER.warning("Force-killing vLLM-Omni server")
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                process.kill()
            process.wait(timeout=30)

    _SERVER_PROCESS = None
    _SERVER_MODEL = None
    _SERVER_ARGS = None


def _wait_for_server_ready(process: subprocess.Popen[str]) -> None:
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    last_error = "server did not respond yet"

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM-Omni server exited with code {process.returncode}\n{_server_log_excerpt()}"
            )

        try:
            response = requests.get(HEALTH_URL, timeout=5)
            if response.status_code == 200:
                models_response = requests.get(MODELS_URL, timeout=5)
                if models_response.status_code == 200:
                    return
                last_error = f"/v1/models returned HTTP {models_response.status_code}"
            else:
                last_error = f"/health returned HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)

        time.sleep(2)

    raise RuntimeError(
        f"Timed out waiting for vLLM-Omni server after {STARTUP_TIMEOUT_SECONDS}s: {last_error}\n"
        f"{_server_log_excerpt()}"
    )


def _start_server_locked(config: ResolvedServerConfig) -> None:
    global _SERVER_PROCESS, _SERVER_MODEL, _SERVER_ARGS

    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("OMP_NUM_THREADS", "1")

    _SERVER_LOG_TAIL.clear()
    command = [
        "python3",
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        config.model,
        "--omni",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
        *config.server_args,
    ]
    LOGGER.info("Starting vLLM-Omni server: %s", " ".join(shlex.quote(part) for part in command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )

    if process.stdout is None:
        raise RuntimeError("Failed to capture vLLM-Omni server logs")

    threading.Thread(target=_reader_thread, args=(process.stdout,), daemon=True).start()
    _SERVER_PROCESS = process
    _SERVER_MODEL = config.model
    _SERVER_ARGS = config.server_args
    _wait_for_server_ready(process)


def ensure_server(config: ResolvedServerConfig) -> None:
    with _SERVER_LOCK:
        if (
            _server_process_running()
            and _server_healthy()
            and _SERVER_MODEL == config.model
            and _SERVER_ARGS == config.server_args
        ):
            return

        _stop_server_locked()
        _start_server_locked(config)


def _encode_image(raw_bytes: bytes, output_format: str) -> tuple[str, int, int]:
    with Image.open(io.BytesIO(raw_bytes)) as image:
        width, height = image.size
        chosen = output_format.upper()
        if chosen == "PNG":
            output_bytes = raw_bytes
        elif chosen in {"JPG", "JPEG"}:
            converted = image.convert("RGB")
            buffer = io.BytesIO()
            converted.save(buffer, format="JPEG", quality=95)
            output_bytes = buffer.getvalue()
            chosen = "JPEG"
        else:
            raise ValueError("output_format must be PNG or JPEG")

    return base64.b64encode(output_bytes).decode("utf-8"), width, height


def _request_images(job_input: dict[str, Any], model_name: str) -> dict[str, Any]:
    prompt = str(job_input.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required")

    width, height = _parse_size(job_input)
    request_body: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "n": _coerce_int(job_input.get("n", job_input.get("num_outputs_per_prompt", 1)), name="n"),
        "response_format": "b64_json",
        "size": f"{width}x{height}",
        "num_inference_steps": _coerce_int(
            job_input.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS),
            name="num_inference_steps",
        ),
    }

    true_cfg_scale = job_input.get("true_cfg_scale", job_input.get("cfg_scale", DEFAULT_TRUE_CFG_SCALE))
    if true_cfg_scale is not None:
        request_body["true_cfg_scale"] = _coerce_float(true_cfg_scale, name="true_cfg_scale")

    guidance_scale = job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
    if guidance_scale is not None:
        request_body["guidance_scale"] = _coerce_float(guidance_scale, name="guidance_scale")

    negative_prompt = str(job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)).strip()
    if negative_prompt:
        request_body["negative_prompt"] = negative_prompt

    if "seed" in job_input and job_input["seed"] is not None:
        request_body["seed"] = _coerce_int(job_input["seed"], name="seed")

    request_started = time.time()
    response = requests.post(
        IMAGES_URL,
        json=request_body,
        timeout=(10, REQUEST_TIMEOUT_SECONDS),
    )

    if response.status_code >= 400:
        raise RuntimeError(f"vLLM-Omni images API HTTP {response.status_code}: {response.text}")

    payload = response.json()
    items = payload.get("data")
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"vLLM-Omni images API returned no images: {payload}")

    output_format = str(job_input.get("output_format", "PNG")).upper()
    images: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("b64_json"), str):
            raise RuntimeError(f"Unexpected image payload at index {index}: {item}")
        raw_bytes = base64.b64decode(item["b64_json"])
        encoded, image_width, image_height = _encode_image(raw_bytes, output_format)
        images.append(
            {
                "index": index,
                "width": image_width,
                "height": image_height,
                "format": output_format if output_format != "JPG" else "JPEG",
                "base64": encoded,
            }
        )

    return {
        "model": model_name,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_images": len(images),
        "num_inference_steps": request_body["num_inference_steps"],
        "true_cfg_scale": request_body.get("true_cfg_scale"),
        "guidance_scale": request_body.get("guidance_scale"),
        "seed": request_body.get("seed"),
        "latency_seconds": round(time.time() - request_started, 2),
        "image_base64": images[0]["base64"],
        "image_format": images[0]["format"],
        "images": images,
    }


def handle_job(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}
    if not isinstance(job_input, dict):
        raise TypeError("input must be an object")

    config = _resolve_server_config(job_input)
    ensure_server(config)

    if job_input.get("warmup"):
        return {
            "status": "warmed",
            "model": config.model,
            "server_args": list(config.server_args),
            "server_url": SERVER_URL,
        }

    result = _request_images(job_input, config.model)
    result["server_args"] = list(config.server_args)
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handle_job})
