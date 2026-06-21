"""
honeypot.py — Detect impossible or fraudulent candidate profiles.

The competition dataset contains ~80 "honeypot" candidates with subtly
impossible profiles. If your top-100 includes more than 10% honeypots,
your submission is disqualified.

Each check is a separate function for easy debugging and transparency.
"""

import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)

# Reference date for all "today" calculations
TODAY: date = date.today()


def _parse_date(date_string: str | None) -> date | None:
    """Parse a date string like '2024-03-08' into a date object."""
    if not date_string:
        return None
    try:
        return datetime.strptime(date_string, "%Y-%m-%d").date()
    except ValueError:
        return None


def _has_salary_anomaly(signals: dict) -> bool:
    """Return True if expected salary min is greater than max.

    A real candidate wouldn't list min > max — this is a data integrity flag.
    """
    salary = signals.get("expected_salary_range_inr_lpa", {})
    min_salary = salary.get("min", 0)
    max_salary = salary.get("max", 0)

    # Both must be positive and min must exceed max by more than rounding
    return min_salary > 0 and max_salary > 0 and min_salary > max_salary + 1.0


def _has_impossible_skill_duration(candidate: dict) -> bool:
    """Return True if any skill has been used longer than the candidate's career.

    Example: 5 years total experience but "Python" listed as 84 months (7 years).
    """
    years_of_experience = candidate.get("profile", {}).get("years_of_experience", 0)
    career_months = years_of_experience * 12

    for skill in candidate.get("skills", []):
        skill_months = skill.get("duration_months", 0)
        # Allow a small buffer of 6 months for rounding
        if skill_months > career_months + 6:
            return True

    return False


def _has_expert_skill_with_zero_months(candidate: dict) -> bool:
    """Return True if candidate claims 'expert' proficiency with 0 months of use.

    A genuine expert would have measurable time using the skill.
    """
    expert_zero_count = 0

    for skill in candidate.get("skills", []):
        is_expert = skill.get("proficiency") == "expert"
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        if is_expert and duration == 0 and endorsements == 0:
            expert_zero_count += 1

    # One could be a data error; multiple is a red flag
    return expert_zero_count >= 3


def _has_experience_company_age_conflict(career_history: list[dict]) -> bool:
    """Return True if a role duration is impossibly long.

    Heuristic: if someone claims 8+ years at a company that was founded
    very recently, that's a honeypot signal. We approximate this by checking
    if start_date is implausibly early given the role's duration.

    Specifically: if duration_months reported > actual months since start_date,
    there's a fabrication.
    """
    for role in career_history:
        start = _parse_date(role.get("start_date"))
        reported_duration = role.get("duration_months", 0)

        if start is None or reported_duration == 0:
            continue

        actual_months_elapsed = (TODAY.year - start.year) * 12 + (TODAY.month - start.month)

        # If reported duration is more than 6 months beyond what's possible, flag it
        if reported_duration > actual_months_elapsed + 6:
            return True

    return False


def _has_ghost_profile(signals: dict) -> bool:
    """Return True if all behavioral signals are suspiciously at zero.

    A real active user would have at least some engagement metrics.
    """
    views = signals.get("profile_views_received_30d", 0)
    applications = signals.get("applications_submitted_30d", 0)
    connections = signals.get("connection_count", 0)
    endorsements = signals.get("endorsements_received", 0)
    search_appearances = signals.get("search_appearance_30d", 0)

    # All five being zero simultaneously is extremely unlikely for a real profile
    return views == 0 and applications == 0 and connections == 0 and endorsements == 0 and search_appearances == 0


def _has_too_many_expert_skills(candidate: dict) -> bool:
    """Return True if candidate claims expert in an unrealistically high number of skills.

    Claiming expert-level in 8+ different technology areas with no endorsements
    or assessment scores is a honeypot signal.
    """
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})

    expert_count = sum(
        1 for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert"
    )

    # 8+ expert skills with no platform assessment at all is suspicious
    if expert_count >= 8 and len(assessment_scores) == 0:
        return True

    return False


def detect_honeypot(candidate: dict) -> bool:
    """Return True if this candidate looks like a honeypot profile.

    Runs all individual checks and returns True if any trigger.
    A honeypot candidate will receive a score of 0.0 from the scorer.

    Args:
        candidate: A single parsed candidate dict.

    Returns:
        True if the candidate appears to be a honeypot, False otherwise.
    """
    candidate_id = candidate.get("candidate_id", "UNKNOWN")
    signals = candidate.get("redrob_signals", {})
    career_history = candidate.get("career_history", [])

    checks = {
        "salary_anomaly":              _has_salary_anomaly(signals),
        "impossible_skill_duration":   _has_impossible_skill_duration(candidate),
        "expert_zero_months":          _has_expert_skill_with_zero_months(candidate),
        "company_age_conflict":        _has_experience_company_age_conflict(career_history),
        "ghost_profile":               _has_ghost_profile(signals),
        "too_many_expert_skills":      _has_too_many_expert_skills(candidate),
    }

    triggered = [name for name, result in checks.items() if result]

    if triggered:
        logger.debug("Honeypot detected for %s — triggers: %s", candidate_id, triggered)
        return True

    return False
