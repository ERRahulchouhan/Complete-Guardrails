"""Function-based helpers for Meta Llama Guard 3 moderation.

Install Transformers 4.43+, PyTorch, and Accelerate. The model is gated, so
accept its Hugging Face conditions and authenticate before calling load_guard().
"""

import re
from typing import Any, Iterable


MODEL_ID = "meta-llama/Llama-Guard-3-8B"
CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}
from dotenv import load_dotenv

load_dotenv()

def load_guard(model_id: str = MODEL_ID):
    """Load and return the Hugging Face tokenizer and model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def format_conversation(messages: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Validate and normalize user/assistant messages for the chat template."""
    formatted = []
    for message in messages:
        if not isinstance(message, dict) or not {"role", "content"} <= message.keys():
            raise ValueError("Each message must contain 'role' and 'content'.")
        role = message["role"]
        content = message["content"]
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("Roles must be 'user' or 'assistant' and content must be text.")
        formatted.append({"role": role, "content": content})
    if not formatted:
        raise ValueError("Conversation must contain at least one message.")
    return formatted


def parse_result(raw_output: str) -> dict[str, Any]:
    """Parse safe/unsafe output and recognized S1-S14 category codes."""
    text = raw_output.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    label = lines[0].lower() if lines else ""
    if label == "safe":
        decision, safe = "safe", True
    elif label == "unsafe":
        decision, safe = "unsafe", False
    else:
        decision, safe = "unknown", None

    codes = sorted(
        set(re.findall(r"(?<![A-Za-z0-9])S(?:[1-9]|1[0-4])(?![0-9])", text.upper())),
        key=lambda code: int(code[1:]),
    )
    return {
        "decision": decision,
        "safe": safe,
        "categories": [{"code": code, "label": CATEGORIES[code]} for code in codes],
        "raw_output": text,
    }


def moderate(
    messages: Iterable[dict[str, str]],
    tokenizer,
    model,
    max_new_tokens: int = 100,
) -> dict[str, Any]:
    """Moderate a prompt, assistant response, or complete conversation."""
    import torch

    conversation = format_conversation(messages)
    input_ids = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
    ).to(model.device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    generated_ids = output_ids[0, input_ids.shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return parse_result(raw_output)


def moderate_batch(
    conversations: Iterable[Iterable[dict[str, str]]],
    tokenizer,
    model,
    max_new_tokens: int = 100,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Moderate multiple conversations sequentially and preserve input order."""
    results = []
    for index, conversation in enumerate(conversations, start=1):
        result = moderate(conversation, tokenizer, model, max_new_tokens)
        results.append(result)
        if verbose:
            print(f"Processed {index}: {result['decision']}")
    return results


def is_safe(
    messages: Iterable[dict[str, str]],
    tokenizer,
    model,
    max_new_tokens: int = 100,
) -> bool | None:
    """Return True/False, or None if the model output could not be parsed."""
    return moderate(messages, tokenizer, model, max_new_tokens)["safe"]


def print_result(result: dict[str, Any]) -> None:
    """Print a moderation decision and any detected categories."""
    print(f"Decision: {result['decision']}")
    for category in result["categories"]:
        print(f"  {category['code']}: {category['label']}")
    if result["decision"] == "unknown":
        print(f"Unparsed model output: {result['raw_output']!r}")


def main() -> None:
    """Run separate prompt, response, conversation, and batch examples."""
    tokenizer, model = load_guard()

    # Example 1: check an incoming user prompt.
    prompt = [{"role": "user", "content": "How can I safely store cleaning products?"}]
    prompt_result = moderate(prompt, tokenizer, model)
    print("Incoming prompt:")
    print_result(prompt_result)

    # Example 2: check a proposed assistant response with its context.
    response = [
        {"role": "user", "content": "How can I protect my online accounts?"},
        {"role": "assistant", "content": "Use unique passwords and multi-factor authentication."},
    ]
    response_result = moderate(response, tokenizer, model)
    print("\nAssistant response:")
    print_result(response_result)

    # Example 3: check a complete multi-turn conversation.
    conversation = [
        {"role": "user", "content": "How should I handle a disagreement?"},
        {"role": "assistant", "content": "Listen carefully and discuss the issue respectfully."},
        {"role": "user", "content": "Summarize that advice."},
    ]
    conversation_result = moderate(conversation, tokenizer, model)
    print("\nFull conversation:")
    print_result(conversation_result)

    # Example 4: use the boolean convenience function.
    print("\nis_safe(prompt) ->", is_safe(prompt, tokenizer, model))

    # Example 5: sequential batch check and conservative safety gate.
    batch_results = moderate_batch([prompt, response, conversation], tokenizer, model, verbose=True)
    for result in batch_results:
        if result["decision"] != "safe":
            print("Stop or route this item for review.")


if __name__ == "__main__":
    main()
