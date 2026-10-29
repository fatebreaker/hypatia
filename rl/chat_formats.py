"""Conversation formats for model families other than Qwen3.

Every model must be trained, sampled and scored in ONE format (the Qwen3 case:
harness/agents.py render_prompt, rl/sft_train.py TRAIN_TEMPLATE,
rl/grpo_onpolicy.py RL_TEMPLATE). This module holds the Mistral format used for
the second-family replication and the Llama-3 format used for the fourth.

MISTRAL. Mistral-7B-Instruct-v0.3's own template attaches the system message to
the LAST user turn -- in an agentic episode the task instructions would move to
the latest tool output at every turn. We instead fold the system message into
the FIRST user turn (as earlier Mistral templates did) and otherwise follow the
official format exactly: "<s>[INST] sys\\n\\nuser[/INST] reply</s>[INST] ...[/INST]".
Every Mistral model here -- the untrained base included -- is prompted this way.
"""
from __future__ import annotations

MISTRAL_TRAIN = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{%- elif message['role'] == 'user' %}"
    "{%- if loop.index0 == 1 and messages[0]['role'] == 'system' %}"
    "{{- '[INST] ' + messages[0]['content'] + '\\n\\n' + message['content'] + '[/INST]' }}"
    "{%- else %}"
    "{{- '[INST] ' + message['content'] + '[/INST]' }}"
    "{%- endif %}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- ' ' }}{% generation %}{{- message['content'] | trim + eos_token }}{% endgeneration %}"
    "{%- endif %}"
    "{%- endfor %}"
)


def render_mistral(messages: list[dict]) -> str:
    """MISTRAL_TRAIN plus the generation point, WITHOUT the leading <s>: the
    server's tokenizer adds BOS to a /completions prompt, and a second one would
    put the model off its format."""
    out, sys_msg = [], None
    for i, m in enumerate(messages):
        if m["role"] == "system":
            sys_msg = m["content"]
        elif m["role"] == "user":
            body = (sys_msg + "\n\n" + m["content"]) if (sys_msg is not None and i == 1) else m["content"]
            out.append(f"[INST] {body}[/INST]")
        elif m["role"] == "assistant":
            out.append(" " + m["content"].strip() + "</s>")
    return "".join(out)


# LLAMA-3. Llama-3.1-8B-Instruct's header format without the template's date preamble (which would
# otherwise differ between the day a trace was trained and the day it is scored):
# "<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>...". Every Llama
# model here -- the untrained base included -- is prompted this way (agent spec :ltrain).
LLAMA3_TRAIN = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
    "{%- if message['role'] == 'assistant' %}"
    "{% generation %}{{- message['content'] | trim + '<|eot_id|>' }}{% endgeneration %}"
    "{%- else %}"
    "{{- message['content'] | trim + '<|eot_id|>' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}"
)


def render_llama3(messages: list[dict]) -> str:
    """LLAMA3_TRAIN plus the generation point, WITHOUT the leading <|begin_of_text|>, which the
    server's tokenizer adds to a /completions prompt."""
    out = [f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>" for m in messages]
    return "".join(out) + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def family(model_path: str) -> str:
    """'mistral', 'llama' or 'qwen', from a model path or hub id."""
    p = str(model_path).lower()
    return "mistral" if "mistral" in p else "llama" if "llama" in p else "qwen"
