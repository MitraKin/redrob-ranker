"""
scorer.py — Combine all feature scores into one composite score per candidate.

This is the "brain" of the ranker. It calls the individual feature functions
and applies the configured weights from config.py.

All scores are in [0, 1]. The final score is also in [0, 1].
"""

import logging

from ranker.config import WEIGHTS
from ranker.features import (
    behavioral_score,
    career_quality_score,
    experience_fit_score,
    logistics_score,
    skill_overlap_score,
)
from ranker.honeypot import detect_honeypot

logger = logging.getLogger(__name__)


def compute_score(candidate: dict, semantic_score: float) -> dict:
    """Compute the full composite score for a single candidate.

    Args:
        candidate:      The parsed candidate dict.
        semantic_score: MiniLM cosine similarity score (0–1) from precompute.py.

    Returns:
        A dict with keys:
          - 'total_score':    float in [0, 1], the final composite score
          - 'is_honeypot':    bool
          - 'sub_scores':     dict of each component score (for debugging / reasoning)
    """
    # Honeypot check first — disqualified candidates get score 0.0
    is_honeypot = detect_honeypot(candidate)

    if is_honeypot:
        logger.debug("Candidate %s flagged as honeypot → score = 0.0", candidate.get("candidate_id"))
        return {
            "total_score": 0.0,
            "is_honeypot": True,
            "sub_scores": {
                "semantic":   0.0,
                "career":     0.0,
                "behavioral": 0.0,
                "experience": 0.0,
                "logistics":  0.0,
            },
        }

    # Compute each component score
    sub_scores = {
        "semantic":   semantic_score,
        "career":     career_quality_score(candidate),
        "behavioral": behavioral_score(candidate),
        "experience": experience_fit_score(candidate),
        "logistics":  logistics_score(candidate),
    }

    # Weighted sum using weights from config.py
    total = sum(WEIGHTS[component] * score for component, score in sub_scores.items())

    return {
        "total_score": round(total, 6),
        "is_honeypot": False,
        "sub_scores":  sub_scores,
    }
