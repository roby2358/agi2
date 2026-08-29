"""Extract dialogue from the Project Gutenberg complete Shakespeare.

Reads samples/shakespeare_complete.txt and writes samples/shakespeare_dialogue.txt,
with each speech stripped of its speaker tag and separated by <|endoftext|>
(a single token in the GPT-2 BPE vocabulary).

The Gutenberg edition formats a speech as a paragraph whose first line is an
all-caps speaker tag ending in a period (e.g. "HORATIO." or "FIRST GAOLER."),
followed by the speech text. Stage directions appear as indented paragraphs or
inline in [_..._] / (_..._) markers. This is a best-effort extraction, not a
perfect one.
"""

import re
from pathlib import Path

SOURCE = Path("samples/shakespeare_complete.txt")
DEST = Path("samples/shakespeare_dialogue.txt")

# An all-caps name (spaces/hyphens/apostrophes allowed) ending in a period,
# optionally followed by the start of the speech on the same line.
SPEAKER_RE = re.compile(r"^([A-Z][A-Z’'\- ]*[A-Z]|[A-Z])\.\s*(.*)$")

# Inline stage directions, possibly spanning lines.
DIRECTION_RE = re.compile(r"\[_.*?\]|\(_.*?\)", re.DOTALL)


def clean_speech(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = DIRECTION_RE.sub("", text)
    text = text.replace("_", "")  # leftover italics markers (songs, letters)
    # Collapse whitespace left behind by removed directions, but keep line breaks.
    cleaned = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in cleaned if line)


def extract(source: Path, dest: Path) -> int:
    turns = 0
    speech: list[str] | None = None
    in_personae = False

    with source.open(encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:

        def flush() -> None:
            nonlocal speech, turns
            if speech is not None:
                text = clean_speech(speech)
                if text:
                    fout.write(f"{text}<|endoftext|>\n")
                    turns += 1
            speech = None

        for raw in fin:
            line = raw.rstrip("\n")
            if not line.strip():
                flush()
                continue
            stripped = line.strip()
            if stripped.lower().startswith(("dramatis person", "persons represented")):
                in_personae = True
            elif stripped.split()[0] in ("ACT", "SCENE", "PROLOGUE", "INDUCTION"):
                in_personae = False
            match = SPEAKER_RE.match(line)
            if in_personae:
                flush()
            elif match and match.group(1).split()[0] not in ("ACT", "SCENE"):
                flush()
                speech = []
                if match.group(2):
                    speech.append(match.group(2))
            elif speech is not None:
                # Indented lines inside a speech are stage directions.
                if line.startswith(" "):
                    continue
                speech.append(line)
        flush()

    return turns


if __name__ == "__main__":
    count = extract(SOURCE, DEST)
    print(f"Wrote {count} turns to {DEST}")
