"""Final-answer extraction used for majority voting."""

from __future__ import annotations


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces), or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        ch = text[i]
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return "".join(out).strip()
        out.append(ch)
        i += 1
    return None


def answer_key(text: str | None) -> str | None:
    """Voting key: the last boxed answer with all whitespace removed."""
    box = last_boxed(text or "")
    return None if box is None else "".join(box.split())
