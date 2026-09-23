"""Local Chinese text embeddings through ONNX Runtime.

Runs BAAI/bge-small-zh-v1.5 with `onnxruntime` + `tokenizers`, so the project
needs neither torch (~2 GB) nor an embedding API, and recall works offline. Model
files are downloaded on first use into `data/models/` and are gitignored.

BGE pooling: the sentence vector is the CLS token of the last hidden state,
L2-normalised. Cosine similarity is then a plain dot product.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_NAME = "bge-small-zh-v1.5"
MODEL_REPO = "Xenova/bge-small-zh-v1.5"
HF_ENDPOINT = "https://hf-mirror.com"

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / MODEL_NAME
ONNX_RELPATH = "onnx/model_quantized.onnx"  # int8, 24 MB instead of 95 MB, same ranking quality
SIDECAR_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

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


def _download(name: str, target: Path) -> None:
    """Fetch one model file, resuming a partial download.

    The fetch shells out to curl: hf-mirror rejects Python's TLS handshake (urllib,
    httpx and requests all die with UNEXPECTED_EOF_WHILE_READING) while curl gets
    through, and curl's `-C -` resumes an interrupted transfer correctly.
    """
    if shutil.which("curl") is None:
        raise RuntimeError(f"curl is required to download {name}; install it or vendor the model")

    url = f"{HF_ENDPOINT}/{MODEL_REPO}/resolve/main/{name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        result = subprocess.run(
            ["curl", "-sSL", "-C", "-", "--max-time", "120", "-o", str(target), url],
            capture_output=True,
        )
        # curl exits 0 only after reading the full remaining body, so a zero exit
        # means the file is complete; a dropped connection exits non-zero and the
        # next attempt picks up where this one stopped.
        if result.returncode == 0 and target.exists() and target.stat().st_size > 0:
            return
    raise RuntimeError(f"could not download {name} after 5 attempts")


def ensure_model(model_dir: Path = MODEL_DIR) -> Path:
    """Download the model on demand (called by the index builder, not by requests)."""
    if not model_is_present(model_dir):
        for name in (ONNX_RELPATH, *SIDECAR_FILES):
            target = model_dir / name
            if not target.exists():
                _download(name, target)
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
