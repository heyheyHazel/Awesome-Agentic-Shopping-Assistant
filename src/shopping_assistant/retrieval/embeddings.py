"""Local Chinese text embeddings through ONNX Runtime.

Runs BAAI/bge-small-zh-v1.5 with `onnxruntime` + `tokenizers`, so the project
needs neither torch (~2 GB) nor an embedding API, and recall works offline. Model
files are downloaded on first use into `models/` and are gitignored.

BGE pooling: the sentence vector is the CLS token of the last hidden state,
L2-normalised. Cosine similarity is then a plain dot product.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from shopping_assistant.settings import get_settings

MODEL_NAME = "bge-small-zh-v1.5"
MODEL_REPO = "Xenova/bge-small-zh-v1.5"
# The model is published on Hugging Face, which is unreachable from some networks.
# Both of these mirror it, and they fail independently, so both are tried.
MODEL_HOSTS = (
    "https://hf-mirror.com",
    "https://aifasthub.com",
)
MODEL_URL = "{host}/" + MODEL_REPO + "/resolve/main/{name}"
ATTEMPTS_PER_SOURCE = 3

MODEL_DIR = get_settings().models_dir / MODEL_NAME
ONNX_RELPATH = "onnx/model_quantized.onnx"  # int8, 24 MB instead of 95 MB, same ranking quality
SIDECAR_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

USER_AGENT = "agentic-shopping-assistant-fetch/1.0"

MAX_TOKENS = 512
BATCH_SIZE = 32


class Encoder:
    """Encodes text into L2-normalised vectors with a local ONNX BGE model."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding()

        options = ort.SessionOptions()
        options.intra_op_num_threads = 0  # let onnxruntime size the pool
        self._session = ort.InferenceSession(
            str(model_dir / ONNX_RELPATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {spec.name for spec in self._session.get_inputs()}
        output = self._session.get_outputs()[0].name
        self._output_name = output
        self.dimension = int(self._session.get_outputs()[0].shape[-1])

    def encode(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """Return an (n, dimension) float32 array of unit-length vectors."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), BATCH_SIZE):
            window = [prefix + text for text in texts[start : start + BATCH_SIZE]]
            encoded = self._tokenizer.encode_batch(window, add_special_tokens=True)

            input_ids = np.array([item.ids for item in encoded], dtype=np.int64)
            attention_mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
            feed = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids)

            hidden = self._session.run([self._output_name], feed)[0]
            # CLS pooling, then L2-normalise so dot products are cosine similarity.
            vectors = hidden[:, 0, :].astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            batches.append(vectors / np.clip(norms, 1e-9, None))

        return np.vstack(batches)


_encoder: Encoder | None = None


def model_is_present(model_dir: Path = MODEL_DIR) -> bool:
    """True when every model file the encoder needs is already on disk."""
    return (model_dir / ONNX_RELPATH).exists() and all(
        (model_dir / name).exists() for name in SIDECAR_FILES
    )


def _mb(size: int) -> str:
    """Bytes as MB, or KB for the small sidecar files, so the log is never "0.0 MB"."""
    return f"{size / 1e6:.1f} MB" if size >= 1e5 else f"{size / 1e3:.0f} KB"


def _fetch_size(url: str) -> int | None:
    """Content-Length reported by a host, or None when it does not say."""
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT}), timeout=30
        ) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:  # noqa: BLE001
        return None


def _fetch_curl(url: str, target: Path) -> bool:
    """One curl attempt; `-C -` resumes an interrupted file instead of restarting it."""
    if shutil.which("curl") is None:
        return False
    result = subprocess.run(  # noqa: PLW1510
        ["curl", "-sSL", "-C", "-", "--max-time", "180", "-o", str(target), url],
        capture_output=True,
    )
    # curl exits 0 only after reading the full remaining body, so a zero exit means
    # the file is complete; a dropped connection exits non-zero and the next attempt
    # picks up where this one stopped.
    return result.returncode == 0 and target.exists()


def _fetch_urllib(url: str, target: Path) -> bool:
    """One urllib attempt, also resuming. Kept because the TLS interference on the
    mirrors is intermittent: when curl is blocked urllib sometimes still gets through."""
    offset = target.stat().st_size if target.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            resuming = offset > 0 and response.status == 206
            with open(target, "ab" if resuming else "wb") as sink:
                while chunk := response.read(1 << 16):
                    sink.write(chunk)
        return True
    except Exception:  # noqa: BLE001
        return False


def _download(name: str, model_dir: Path) -> Path:
    """Fetch one model file, trying every mirror and client before giving up.

    A partial file is always resumed rather than restarted, and the size is checked
    against the host's Content-Length when one is offered, so a half-written file is
    never mistaken for a complete one.
    """
    target = model_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)

    for host in MODEL_HOSTS:
        url = MODEL_URL.format(host=host, name=name)
        expected = _fetch_size(url)
        if target.exists() and expected is not None and target.stat().st_size == expected:
            print(f"    {name}: cached")
            return target

        for attempt in range(1, ATTEMPTS_PER_SOURCE + 1):
            for client, fetch in (("curl", _fetch_curl), ("urllib", _fetch_urllib)):
                if fetch(url, target):
                    size = target.stat().st_size
                    if expected is None or size == expected:
                        print(f"    {name}: {_mb(size)} via {client} on {host.split('//')[1]}")
                        return target
            done = target.stat().st_size if target.exists() else 0
            total = f"/{_mb(expected)}" if expected else ""
            print(f"    {name}: {_mb(done)}{total} — retrying ({attempt}/{ATTEMPTS_PER_SOURCE} on {host.split('//')[1]})")

    raise RuntimeError(
        f"could not download {name} from any of {', '.join(MODEL_HOSTS)}. "
        "Retry when the network recovers: the partial file is kept and resumed."
    )


def ensure_model(model_dir: Path = MODEL_DIR) -> Path:
    """Download whatever is missing. Safe to re-run: complete files are skipped."""
    for name in (ONNX_RELPATH, *SIDECAR_FILES):
        if not (model_dir / name).exists():
            _download(name, model_dir)
    return model_dir


def get_encoder() -> Encoder:
    """Lazily build the process-wide encoder.

    Never downloads: a request path must not stall on a 24 MB fetch. Run
    `scripts/build_index.py`, which calls `ensure_model()` first, to populate the cache.
    """
    global _encoder
    if _encoder is None:
        if not model_is_present():
            raise RuntimeError(f"embedding model missing at {MODEL_DIR}; run: python scripts/build_index.py")
        _encoder = Encoder()
    return _encoder
