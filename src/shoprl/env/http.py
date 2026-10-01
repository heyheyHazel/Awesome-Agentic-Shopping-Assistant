"""Client for the upstream ShopSimulator Flask service.

Same method names and payloads as :class:`shoprl.env.local.LocalShopEnv`, so
switching between the reference environment and a fast local one is a config
change. Use this one when a number has to be comparable with published results.
"""

from __future__ import annotations

from typing import Any

import httpx


class RemoteShopEnv:
    def __init__(self, base_url: str, *, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self.env_idx: int | None = None
        self.session_id = ""

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(self.base_url, json=payload)
        response.raise_for_status()
        body = response.json()
        result = body.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"malformed environment response: {body}")  # noqa: TRY004
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        return result

    def reset(self, task_id: int, session_id: str) -> dict[str, Any]:
        result = self._call(
            {"action": "reset", "idx": task_id, "rollout_session_id": session_id}
        )
        self.env_idx = int(result["env_idx"])
        self.session_id = session_id
        return result

    def interact(self, action: str, session_id: str) -> dict[str, Any]:
        return self._call(
            {
                "action": "interact",
                "env_idx": self.env_idx,
                "response": action,
                "rollout_session_id": session_id,
            }
        )

    def release(self, session_id: str | None = None) -> None:
        if self.env_idx is None:
            return
        try:
            self._call(
                {
                    "action": "release_one",
                    "env_idx": self.env_idx,
                    "rollout_session_id": session_id or self.session_id,
                }
            )
        finally:
            self.env_idx = None

    def status(self) -> dict[str, Any]:
        return self._call({"action": "status"})

