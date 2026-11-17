# -*- coding: utf-8 -*-
"""Write bge.yaml and adapt SeCom's utils.py for Windows/DeepSeek.

Usage:
    python adapt_secom.py D:\\longmemeval
"""

import os
import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python adapt_secom.py <WORK>")

    work = Path(sys.argv[1]).resolve()
    secom = work / "SeCom"
    utils = secom / "secom" / "utils.py"
    cfg = secom / "secom" / "configs" / "bge.yaml"

    if not utils.is_file():
        raise SystemExit("missing: {}".format(utils))

    model = os.environ.get("SEGMENT_MODEL", "deepseek-v4-free")
    cfg.write_text(
        "segmentor:\n"
        "  segment_model: {}\n"
        "  prompt_path: instructions/segment_with_exchange_number.md\n"
        "  incremental_prompt_path: instructions/segment_incremental.md\n"
        "compressor:\n"
        "  compress_model: microsoft/llmlingua-2-xlm-roberta-large-meetingbank\n"
        "retriever:\n"
        "  storage: FAISS\n"
        "  embedding_model: BAAI/bge-base-en-v1.5\n"
        "  device_map: cuda\n".format(model),
        encoding="utf-8",
    )
    print("wrote", cfg)

    source = utils.read_text(encoding="utf-8")

    # Remove the hard top-level vllm import. LocalLLM is unused by this harness,
    # but keep it lazy-importable rather than deleting the class.
    source = re.sub(
        r"^from vllm import LLM, SamplingParams[ \t]*\n",
        "",
        source,
        flags=re.M,
    )
    source = source.replace(
        "        self.model = LLM(model=model_name_or_path)",
        "        from vllm import LLM\n"
        "        self.model = LLM(model=model_name_or_path)",
    )

    needle = 'create_kwargs["reasoning_effort"] = "none"'
    if needle not in source:
        old_call = """                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                )"""
        new_call = """                create_kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                create_kwargs["extra_body"] = {
                    "thinking": {"type": "disabled"},
                    "reasoning_effort": "none",
                }
                completion = self.client.chat.completions.create(**create_kwargs)"""

        if old_call in source:
            source = source.replace(old_call, new_call, 1)
        elif 'create_kwargs["extra_body"]' not in source:
            raise SystemExit("cannot find OpenAI completion call to patch")

    utils.write_text(source, encoding="utf-8")
    print("patched", utils)

    # SeCom reads this path via os.path.expanduser("~/dot_env/openai.env").
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com/v1")

    env_dir = Path.home() / "dot_env"
    env_dir.mkdir(parents=True, exist_ok=True)
    env_file = env_dir / "openai.env"
    env_file.write_text(
        "OPENAI_API_KEY={}\nOPENAI_API_BASE={}\n".format(api_key, api_base),
        encoding="utf-8",
    )
    print("wrote", env_file)

    import py_compile

    py_compile.compile(str(utils), doraise=True)
    print("utils.py compile OK")


if __name__ == "__main__":
    main()
