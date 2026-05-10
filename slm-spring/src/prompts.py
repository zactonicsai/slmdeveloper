"""Prompt templates. Keep these tight — narrow templates are the single most
effective tool for reducing hallucination in a small fine-tuned model."""

from typing import List, Dict


SYSTEM_INSTRUCTION = (
    "You are a Java 21 + Spring Boot 3.5.13 code generator. "
    "You output ONLY valid Java source code, no explanations, no markdown fences. "
    "Use Lombok (@Data, @Builder, @RequiredArgsConstructor, @Slf4j) and Jakarta Validation "
    "(jakarta.validation.constraints.*). Use jakarta.* not javax.*. "
    "Constructor injection only — no @Autowired on fields. "
    "Wrap every controller method response in ResponseEntity."
)


def build_training_prompt(instruction: str, output: str) -> str:
    """Used during fine-tuning. Includes the target output."""
    return (
        f"### System:\n{SYSTEM_INSTRUCTION}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}\n"
    )


def build_inference_prompt(instruction: str, retrieved_examples: List[Dict]) -> str:
    """Used at inference. Prepends the top-k retrieved examples as grounded context.

    `retrieved_examples` is a list of {"content": str, "metadata": {...}} dicts
    coming back from ChromaDB.
    """
    context_block = "\n".join(
        f"--- Reference example ({ex['metadata'].get('artifact_type', 'unknown')}: "
        f"{ex['metadata'].get('entity', '?')}) ---\n{ex['content']}"
        for ex in retrieved_examples
    )

    return (
        f"### System:\n{SYSTEM_INSTRUCTION}\n\n"
        f"### Reference patterns (treat as the ground truth for style and imports):\n"
        f"{context_block}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )
