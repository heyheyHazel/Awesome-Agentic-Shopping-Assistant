"""Rollout engines: the policy under training, behind the harness backend API.

Both engines return the sampled token ids and their log-probabilities. That is
what lets RL use the tokens the policy actually produced instead of
re-tokenising text and silently training on a different sequence.
"""

from __future__ import annotations

from typing import Any

from shoprl.harness.types import Message, ModelResponse
from shoprl.train.parsing import parse_tool_calls, strip_tool_calls


def render_prompt(tokenizer, messages: list[Message], tools: list[dict[str, Any]] | None):
    """Tokenise a conversation with the model's own chat template."""
    conversation = [message.as_dict() for message in messages]
    return tokenizer.apply_chat_template(
        conversation,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )


class HFRolloutEngine:
    """``transformers`` sampler. Always available; the portable default."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        on_token=None,
    ) -> ModelResponse:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        inputs = render_prompt(self.tokenizer, messages, tools).to(self.device)
        prompt_length = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=1.0,
                max_new_tokens=max_tokens or self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = output.sequences[0][prompt_length:]
        scores = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )[0]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return ModelResponse(
            text=strip_tool_calls(text),
            tool_calls=parse_tool_calls(text),
            token_ids=[int(token) for token in generated],
            logprobs=[float(value) for value in scores],
            finish_reason="stop",
        )


class VLLMRolloutEngine:
    """``vLLM`` sampler for throughput; imported lazily so CPU hosts still import."""

    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int = 256,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 18432,
        **engine_kwargs: Any,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            **engine_kwargs,
        )
        self.max_new_tokens = max_new_tokens
        self._sampling = None

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        on_token=None,
    ) -> ModelResponse:
        from vllm import SamplingParams

        prompt = self.tokenizer.apply_chat_template(
            [message.as_dict() for message in messages],
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling = SamplingParams(
            temperature=max(temperature, 1e-5) if temperature > 0 else 0.0,
            top_p=1.0,
            max_tokens=max_tokens or self.max_new_tokens,
            seed=seed,
            logprobs=1,
        )
        result = self.llm.generate([prompt], sampling, use_tqdm=False)[0].outputs[0]
        return ModelResponse(
            text=strip_tool_calls(result.text),
            tool_calls=parse_tool_calls(result.text),
            token_ids=list(result.token_ids),
            logprobs=[next(iter(step.values()))[0] if step else 0.0 for step in (result.logprobs or [])],
            finish_reason=str(result.finish_reason or "stop"),
        )

    def sleep(self) -> None:
        """Release KV cache so the trainer can use the whole card."""
        if hasattr(self.llm, "sleep"):
            self.llm.sleep(level=1)

    def wake(self) -> None:
        if hasattr(self.llm, "wake_up"):
            self.llm.wake_up()
