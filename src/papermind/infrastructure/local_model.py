"""Local Qwen2.5-Coder-7B loader with bitsandbytes NF4 quantization.

NF4 (NormalFloat4) quantization exploits the fact that neural network weights
are approximately normally distributed. It defines 16 quantization bins at
equal-probability quantiles of N(0,1):

    [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
     0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]

For each block of 64 weights:
  1. Compute absmax = max(|weights|) for the block
  2. Normalize: w_norm = w / absmax
  3. Map each normalized weight to the nearest NF4 quantile
  4. Store as a 4-bit index (two per byte) + one fp32 absmax per block

Double quantization (QLoRA innovation) additionally quantizes the per-block
absmax values to 8-bit, saving ~0.4 bits per parameter.

Dequantization at inference: value = NF4_lookup[index] * absmax
Mixed-precision matmul: output = X_fp16 @ W_dequantized in compute_dtype,
avoiding materializing the full FP16 weight matrix via fused kernels.

Memory budget for Qwen2.5-Coder-7B at NF4:
  - 7.6B params × 4 bits = ~3.5 GB (weights)
  - Double quant absmax overhead: ~50 MB
  - KV cache at 16K context: ~0.5 GB
  - Total: ~4.0-4.5 GB, leaving ~7.5 GB free on 12 GB VRAM
"""

from collections.abc import Iterator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

from papermind.config import get_settings


def _build_quantization_config(
    quantization: str = "nf4",
    double_quant: bool = True,
    compute_dtype: str = "bfloat16",
) -> BitsAndBytesConfig | None:
    """Build the bitsandbytes quantization config.

    NF4 quantization:
      - bnb_4bit_quant_type="nf4": Use NormalFloat4 bins (vs. uniform "fp4")
      - bnb_4bit_use_double_quant=True: Quantize the absmax scale factors to
        8-bit, saving ~0.4 bits/param (~200MB on a 7B model)
      - bnb_4bit_compute_dtype: The dtype for the dequantized matmul.
        bfloat16 is preferred — same range as fp32 so no overflow on large
        activations, and natively supported on Ampere+. fp16 is slightly
        faster on older GPUs but risks overflow in attention layers.

    INT8 quantization (LLM.int8()):
      - Uses absmax quantization per row/column
      - Detects "outlier" features (>6σ) and keeps them in fp16
      - ~7.5 GB for 7B model (vs ~3.5 GB for NF4)
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dt = dtype_map.get(compute_dtype, torch.bfloat16)

    if quantization == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_compute_dtype=compute_dt,
        )
    elif quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "none":
        return None
    else:
        raise ValueError(f"Unknown quantization type: {quantization}")


class LocalModel:
    """Manages local Qwen2.5-Coder-7B with 4-bit quantization.

    Usage:
        model = LocalModel()
        model.load()  # Downloads + quantizes on first run

        # Single generation
        response = model.generate("Write a Python function for binary search")

        # Chat with message history
        response = model.chat([
            {"role": "user", "content": "Explain attention mechanisms"}
        ])

        # Streaming generation
        for token in model.generate_stream("Write a tokenizer"):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model_name: str | None = None,
        quantization: str | None = None,
        double_quant: bool | None = None,
        compute_dtype: str | None = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.llm.local_model
        self._quantization = quantization or settings.llm.quantization
        self._double_quant = double_quant if double_quant is not None else settings.llm.double_quant
        self._compute_dtype = compute_dtype or settings.llm.compute_dtype
        self._max_new_tokens = settings.llm.max_new_tokens
        self._temperature = settings.llm.temperature
        self._top_p = settings.llm.top_p
        self._repetition_penalty = settings.llm.repetition_penalty

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    def load(self) -> None:
        """Load the model and tokenizer with quantization.

        First call downloads the model (~4 GB for NF4). Subsequent calls
        are fast since HuggingFace caches weights locally.
        """
        if self._model is not None:
            return

        print(f"Loading {self.model_name} with {self._quantization} quantization...")

        quant_config = _build_quantization_config(
            self._quantization, self._double_quant, self._compute_dtype
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Clear any stale CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Loading a 7B model with NF4 on 12GB VRAM requires careful memory management.
        #
        # The problem: transformers v5+ uses concurrent weight loading that
        # materializes weights in the compute dtype on GPU BEFORE quantizing.
        # Multiple concurrent shards can exhaust VRAM even though the final
        # quantized model only needs ~4GB.
        #
        # The solution: set CUDA_VISIBLE_DEVICES and max_memory to tightly
        # control GPU allocation, and use torch_dtype=float16 to keep the
        # loading intermediate smaller (float16 vs bfloat16 same size but
        # more compatible). The key is max_memory leaving enough headroom
        # for the concurrent materializer's peak usage.
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # Serialize weight loading to avoid concurrent GPU memory spikes.
        # The transformers concurrent materializer can use N × shard_size
        # transiently, which exceeds 12GB for a 7B model.
        import os
        os.environ.setdefault("TRANSFORMERS_LOADING_WORKERS", "1")

        load_kwargs: dict = {
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "max_memory": {0: f"{gpu_total_gb - 3.5:.0f}GiB", "cpu": "24GiB"},
            "dtype": torch.float16,
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )

        self._report_memory()

    def _report_memory(self) -> None:
        """Print GPU memory usage after loading."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(
                f"  VRAM: {allocated:.1f} GB allocated, "
                f"{reserved:.1f} GB reserved, "
                f"{total:.1f} GB total"
            )
            print(f"  Free: {total - reserved:.1f} GB")

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a completion from a prompt using the chat template.

        Wraps the prompt in Qwen2.5's chat format:
            <|im_start|>system\n{system}<|im_end|>\n
            <|im_start|>user\n{prompt}<|im_end|>\n
            <|im_start|>assistant\n

        This is the ChatML template that Qwen2.5-Instruct models expect.
        The tokenizer.apply_chat_template() handles this automatically.
        """
        messages = self._build_messages(prompt, system)
        return self._generate_from_messages(messages, max_new_tokens, temperature)

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Multi-turn chat completion. Messages are [{"role": ..., "content": ...}]."""
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)
        return self._generate_from_messages(all_messages, max_new_tokens, temperature)

    def generate_stream(
        self,
        prompt: str,
        system: str = "",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream tokens one at a time using TextIteratorStreamer."""
        import threading

        messages = self._build_messages(prompt, system)
        input_ids = self._apply_template(messages)
        gen_kwargs = self._gen_kwargs(max_new_tokens, temperature)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(
            target=self.model.generate, kwargs={**gen_kwargs, "input_ids": input_ids}
        )
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream a multi-turn chat completion token by token."""
        import threading

        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        input_ids = self._apply_template(all_messages)
        gen_kwargs = self._gen_kwargs(max_new_tokens, temperature)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(
            target=self.model.generate, kwargs={**gen_kwargs, "input_ids": input_ids}
        )
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    def _build_messages(
        self, prompt: str, system: str = ""
    ) -> list[dict[str, str]]:
        """Build the message list for the chat template."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        else:
            messages.append({
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            })
        messages.append({"role": "user", "content": prompt})
        return messages

    def _apply_template(
        self, messages: list[dict[str, str]]
    ) -> torch.Tensor:
        """Apply the chat template and tokenize.

        Qwen2.5 uses the ChatML format:
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            Hello<|im_end|>
            <|im_start|>assistant

        The tokenizer handles this via apply_chat_template(), which:
          1. Wraps each message with <|im_start|>{role}\n{content}<|im_end|>\n
          2. Appends <|im_start|>assistant\n to prompt generation
          3. Returns token ids ready for model.generate()
        """
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt")
        return inputs.input_ids.to(self.model.device)

    def _gen_kwargs(
        self,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        """Build generation kwargs."""
        temp = temperature if temperature is not None else self._temperature
        kwargs: dict = {
            "max_new_tokens": max_new_tokens or self._max_new_tokens,
            "repetition_penalty": self._repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if temp < 0.01:
            # Greedy decoding
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temp
            kwargs["top_p"] = self._top_p
        return kwargs

    def _generate_from_messages(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Core generation: template → tokenize → generate → decode."""
        input_ids = self._apply_template(messages)
        gen_kwargs = self._gen_kwargs(max_new_tokens, temperature)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids=input_ids, **gen_kwargs)

        # Slice off the prompt tokens to get only the generated response
        generated_ids = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def unload(self) -> None:
        """Free GPU memory by deleting model and clearing CUDA cache."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def vram_usage(self) -> dict:
        """Return current VRAM usage stats."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "free_gb": round(
                (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())
                / 1024**3,
                2,
            ),
        }
