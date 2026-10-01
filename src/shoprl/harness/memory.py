"""Session memory: a small, deterministic, model-writable scratchpad.

Long shopping episodes lose the constraints a shopper stated early. Memory keeps
them as pinned text so the context policy can prune the transcript without
pruning the requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MemoryNote:
    content: str
    key: str = ""
    turn: int = 0


@dataclass(slots=True)
class Memory:
    """Bounded key/value notes rendered into a single pinned system block."""

    max_notes: int = 12
    max_chars: int = 1200
    notes: list[MemoryNote] = field(default_factory=list)

    def remember(self, content: str, *, key: str = "", turn: int = 0) -> None:
        content = " ".join(content.split())
        if not content:
            return
        if key:
            self.notes = [note for note in self.notes if note.key != key]
        self.notes.append(MemoryNote(content=content, key=key, turn=turn))
        del self.notes[: max(0, len(self.notes) - self.max_notes)]

    def forget(self, key: str) -> None:
        self.notes = [note for note in self.notes if note.key != key]

    def render(self) -> str:
        """Return the pinned block, or "" when there is nothing to pin."""
        lines: list[str] = []
        used = 0
        for note in reversed(self.notes):
            line = f"- {note.content}"
            if used + len(line) > self.max_chars:
                break
            lines.append(line)
            used += len(line)
        if not lines:
            return ""
        return "Your notes from earlier in this session:\n" + "\n".join(reversed(lines))

