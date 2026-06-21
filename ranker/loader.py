"""
loader.py — Load and stream candidates from a JSONL file.

Streams one record at a time so we never load all 487 MB into memory at once.
"""

import json
import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


def stream_candidates(jsonl_path: Path) -> Generator[dict, None, None]:
    """Yield one candidate dict at a time from a JSONL file.

    Skips blank lines and logs (but does not crash on) malformed JSON.

    Args:
        jsonl_path: Path to the candidates.jsonl file.

    Yields:
        A single parsed candidate dict.
    """
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue  # skip blank lines

            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning("Skipping malformed JSON on line %d: %s", line_number, error)


def build_candidate_text(candidate: dict) -> str:
    """Combine a candidate's text fields into one string for embedding.

    Pulls from: headline, summary, career descriptions, skill names.
    The richer the text, the better the semantic match with the JD.

    Args:
        candidate: A single parsed candidate dict.

    Returns:
        A single concatenated text string.
    """
    parts: list[str] = []

    profile = candidate.get("profile", {})

    # Headline and summary carry role intent
    if profile.get("headline"):
        parts.append(profile["headline"])

    if profile.get("summary"):
        parts.append(profile["summary"])

    # Career descriptions are the richest source of actual work done
    for role in candidate.get("career_history", []):
        if role.get("title"):
            parts.append(role["title"])
        if role.get("description"):
            parts.append(role["description"])

    # Skill names add vocabulary for matching
    skill_names = [skill["name"] for skill in candidate.get("skills", []) if skill.get("name")]
    if skill_names:
        parts.append(" ".join(skill_names))

    # Certification names also help
    cert_names = [cert["name"] for cert in candidate.get("certifications", []) if cert.get("name")]
    if cert_names:
        parts.append(" ".join(cert_names))

    return " ".join(parts)


def count_candidates(jsonl_path: Path) -> int:
    """Count total candidates in the file (for progress bars).

    Args:
        jsonl_path: Path to the candidates.jsonl file.

    Returns:
        Number of non-blank lines.
    """
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count
