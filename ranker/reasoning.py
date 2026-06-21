"""
reasoning.py — Generate a 1–2 sentence reasoning string for each ranked candidate.

The reasoning must:
  - Reference specific facts from the candidate's actual profile (no hallucination)
  - Mention at least one concern when relevant
  - Vary by profile type (don't template everything the same way)
  - Match the tone of the rank (rank-1 sounds strong; rank-95 sounds marginal)

Each helper builds one "block" of a sentence. The main function assembles them.
"""

from ranker.config import PREFERRED_LOCATIONS, STALENESS_CUTOFF_DAYS
from ranker.features import extract_key_facts


def _format_years(years: float) -> str:
    """Format years of experience cleanly: '6 yrs', '6.5 yrs'."""
    if years == int(years):
        return f"{int(years)} yrs"
    return f"{years:.1f} yrs"


def _format_notice(days: int) -> str:
    """Format notice period as a human-readable string."""
    if days == 0:
        return "immediately available"
    if days <= 30:
        return f"{days}-day notice"
    return f"{days}-day notice period"


def _build_strength_phrase(facts: dict, sub_scores: dict) -> str:
    """Pick the single strongest positive aspect to lead the reasoning with."""

    # Identify the best sub-score (excluding logistics which is rarely the lead)
    scoreable = {
        k: v for k, v in sub_scores.items()
        if k not in ("logistics",)
    }
    best_component = max(scoreable, key=scoreable.get)

    if best_component == "semantic":
        title = facts["best_title"]
        company = facts["best_company"]
        yrs = _format_years(facts["years_exp"])
        return f"{title} at {company} ({yrs})"

    if best_component == "career":
        company = facts["best_company"]
        title = facts["best_title"]
        return f"{title} background at product company {company}"

    if best_component == "behavioral":
        rate = facts["response_rate"]
        if facts["open_to_work"]:
            return f"actively seeking (response rate {rate:.0%})"
        return f"high engagement on platform (response rate {rate:.0%})"

    if best_component == "experience":
        github = facts["github_score"]
        yrs = _format_years(facts["years_exp"])
        if github != -1 and github >= 50:
            return f"{yrs} experience, strong GitHub activity ({int(github)}/100)"
        return f"{yrs} of hands-on ML/AI experience"

    return f"{_format_years(facts['years_exp'])} experience"


def _build_skills_phrase(facts: dict) -> str:
    """Mention the top 2 relevant skills from the candidate's profile."""
    skills = facts.get("top_skills", [])
    if not skills:
        return ""
    if len(skills) == 1:
        return f"skills include {skills[0]}"
    return f"key skills: {', '.join(skills[:2])}"


def _build_concern_phrase(facts: dict, sub_scores: dict) -> str:
    """Surface one meaningful concern if any — honest reasoning scores higher."""

    # Long notice period
    if facts["notice_days"] > 90:
        return f"concern: long notice period ({facts['notice_days']} days)"

    # Stale profile
    days_active = facts.get("days_since_active")
    if days_active and days_active > STALENESS_CUTOFF_DAYS:
        months_inactive = days_active // 30
        return f"concern: last active {months_inactive} months ago"

    # Low response rate
    if facts["response_rate"] < 0.25:
        return f"low recruiter response rate ({facts['response_rate']:.0%})"

    # Location mismatch
    location = facts.get("location", "").lower()
    in_preferred = any(city in location for city in PREFERRED_LOCATIONS)
    if not in_preferred:
        return f"location ({facts['location']}) may require relocation"

    # Low behavioral engagement
    if sub_scores.get("behavioral", 1.0) < 0.35:
        return "low overall platform engagement"

    return ""  # No meaningful concern to raise


def generate_reasoning(candidate: dict, result: dict, rank: int) -> str:
    """Generate a 1–2 sentence reasoning string for a ranked candidate.

    Args:
        candidate: The parsed candidate dict.
        result:    Output from scorer.compute_score() — includes sub_scores.
        rank:      The final rank (1 = best).

    Returns:
        A 1–2 sentence string, 20–50 words, referencing real profile facts.
    """
    facts = extract_key_facts(candidate)
    sub_scores = result.get("sub_scores", {})

    # Build the core positive phrase
    strength = _build_strength_phrase(facts, sub_scores)
    skills = _build_skills_phrase(facts)
    concern = _build_concern_phrase(facts, sub_scores)
    notice = _format_notice(facts["notice_days"])

    # Assemble the sentence differently based on rank tier
    if rank <= 10:
        # Top-10: lead with full strength, mention notice, add concern only if serious
        sentence = f"{strength}; {skills}; {notice}."
        if concern:
            sentence += f" {concern.capitalize()}."

    elif rank <= 50:
        # Mid-tier: strength + one fact + concern
        sentence = f"{strength}; {skills}."
        if concern:
            sentence += f" Note: {concern}."
        else:
            sentence += f" Available in {notice}."

    else:
        # Bottom tier: honest, acknowledge they're a stretch
        sentence = f"Adjacent fit — {strength}."
        if concern:
            sentence += f" {concern.capitalize()}."
        else:
            sentence += f" Skills overlap is partial; included as best available at this rank."

    # Ensure it doesn't exceed 2 sentences
    parts = sentence.split(". ")
    if len(parts) > 2:
        sentence = ". ".join(parts[:2]) + "."

    return sentence
