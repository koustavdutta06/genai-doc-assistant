import re

_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"reveal your (instructions|prompt|system)",
]

_UNSAFE_KEYWORDS = [
    "bomb", "explosive", "kill someone", "make a weapon",
]

MAX_QUESTION_LENGTH = 2000


def check_input_safety(question: str) -> tuple[bool, str | None]:
    stripped = question.strip()
    if not stripped:
        return False, "Question is empty."
    if len(stripped) > MAX_QUESTION_LENGTH:
        return False, "Question is too long."

    lowered = stripped.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, "Question appears to contain a prompt-injection attempt."

    for keyword in _UNSAFE_KEYWORDS:
        if keyword in lowered:
            return False, "Question requests unsafe content."

    return True, None


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def check_grounding(answer: str, context_chunks: list[str]) -> bool:
    abstain_markers = ["don't have enough information", "do not have enough information", "cannot answer"]
    lowered_answer = answer.lower()
    if any(marker in lowered_answer for marker in abstain_markers):
        return True

    if not context_chunks:
        return False

    context_text = " ".join(context_chunks).lower()
    context_words = set(re.findall(r"\b\w{4,}\b", context_text))

    sentences = _sentences(answer)
    if not sentences:
        return False

    grounded_count = 0
    for sentence in sentences:
        words = set(re.findall(r"\b\w{4,}\b", sentence.lower()))
        if not words:
            grounded_count += 1
            continue
        overlap = len(words & context_words) / len(words)
        if overlap >= 0.3:
            grounded_count += 1

    return (grounded_count / len(sentences)) >= 0.6
