"""
features.py — Extract structured features from a candidate profile.

Each public function takes a candidate dict and returns a float in [0.0, 1.0].
All scoring logic lives here; the scorer.py just combines the outputs.

Design principle: one function = one clear concern.
"""

import logging
import re
from datetime import date, datetime

from ranker.config import (
    CONSULTING_FIRMS,
    IDEAL_YOE_MAX,
    IDEAL_YOE_MIN,
    JD_NICE_TO_HAVE_SKILLS,
    JD_REQUIRED_SKILLS,
    MIN_RESPONSE_RATE,
    NOTICE_ACCEPTABLE_DAYS,
    NOTICE_IDEAL_DAYS,
    NOTICE_OK_DAYS,
    PREFERRED_LOCATIONS,
    PRODUCTION_KEYWORDS,
    SALARY_BAND_MAX_LPA,
    SALARY_BAND_MIN_LPA,
    STALENESS_CUTOFF_DAYS,
    STRONG_TITLE_KEYWORDS,
    WEAK_TITLE_KEYWORDS,
)

logger = logging.getLogger(__name__)

TODAY: date = date.today()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase and strip extra whitespace from text for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _days_since(date_string: str | None) -> int | None:
    """Return the number of days between a date string and today."""
    if not date_string:
        return None
    try:
        past = datetime.strptime(date_string, "%Y-%m-%d").date()
        return (TODAY - past).days
    except ValueError:
        return None


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# Skill matching features
# ---------------------------------------------------------------------------

PROFICIENCY_WEIGHTS: dict[str, float] = {
    "beginner":     0.25,
    "intermediate": 0.50,
    "advanced":     0.75,
    "expert":       1.00,
}


def skill_overlap_score(candidate: dict) -> float:
    """Score how well the candidate's skills match JD requirements.

    Gives full credit for required skills, half credit for nice-to-haves.
    Weights each matched skill by the candidate's proficiency level.

    Returns:
        Float in [0, 1]. 1.0 = perfect match on all required + nice-to-have skills.
    """
    candidate_skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})

    # Build a lookup: normalised skill name → (proficiency_weight, duration_months)
    skill_lookup: dict[str, tuple[float, int]] = {}
    for skill in candidate_skills:
        name = _normalise(skill.get("name", ""))
        proficiency = skill.get("proficiency", "beginner")
        duration = skill.get("duration_months", 0)
        weight = PROFICIENCY_WEIGHTS.get(proficiency, 0.25)
        skill_lookup[name] = (weight, duration)

    total_score = 0.0
    max_possible = 0.0

    # Required skills are worth double
    for jd_skill in JD_REQUIRED_SKILLS:
        jd_skill_lower = _normalise(jd_skill)
        max_possible += 2.0

        for candidate_skill, (prof_weight, duration) in skill_lookup.items():
            if jd_skill_lower in candidate_skill or candidate_skill in jd_skill_lower:
                # Bonus if the candidate has a platform assessment score for this skill
                assessment_bonus = 0.0
                for assessed_skill, assessed_score in assessment_scores.items():
                    if jd_skill_lower in _normalise(assessed_skill):
                        assessment_bonus = (assessed_score / 100) * 0.3
                        break

                # Duration bonus: more months = more confidence (cap at 3 years)
                duration_bonus = min(duration / 36, 1.0) * 0.2

                total_score += 2.0 * (prof_weight + assessment_bonus + duration_bonus)
                break  # only count once per JD skill

    # Nice-to-have skills are worth 1.0
    for jd_skill in JD_NICE_TO_HAVE_SKILLS:
        jd_skill_lower = _normalise(jd_skill)
        max_possible += 1.0

        for candidate_skill, (prof_weight, _) in skill_lookup.items():
            if jd_skill_lower in candidate_skill or candidate_skill in jd_skill_lower:
                total_score += 1.0 * prof_weight
                break

    if max_possible == 0:
        return 0.0

    return _clamp(total_score / max_possible)


# ---------------------------------------------------------------------------
# Career quality features
# ---------------------------------------------------------------------------

def _is_consulting_firm(company_name: str) -> bool:
    """Return True if the company is a known large IT consulting firm."""
    company_lower = _normalise(company_name)
    return any(firm in company_lower for firm in CONSULTING_FIRMS)


def career_quality_score(candidate: dict) -> float:
    """Score the quality and relevance of the candidate's career trajectory.

    Rewards:
    - Experience at product companies (not pure IT services)
    - Stable tenures (not job-hopping every 1.5 years)
    - Titles that match the target role
    - Production deployment language in descriptions

    Penalises:
    - Entire career spent only at consulting firms
    - Very short average tenures (< 15 months)
    - Non-technical current title

    Returns:
        Float in [0, 1].
    """
    career_history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    if not career_history:
        return 0.0

    # --- Product company experience ---
    total_months = 0
    consulting_months = 0
    product_months = 0

    for role in career_history:
        duration = role.get("duration_months", 0)
        company = role.get("company", "")
        total_months += duration

        if _is_consulting_firm(company):
            consulting_months += duration
        else:
            product_months += duration

    # Entirely consulting career is a JD disqualifier
    if total_months > 0 and consulting_months / total_months > 0.95:
        consulting_penalty = 0.1
    elif total_months > 0 and consulting_months / total_months > 0.7:
        consulting_penalty = 0.5
    else:
        consulting_penalty = 1.0

    product_ratio = product_months / max(total_months, 1)
    product_score = _clamp(product_ratio)

    # --- Job stability (tenure) ---
    completed_roles = [r for r in career_history if not r.get("is_current", False)]
    if completed_roles:
        avg_tenure = sum(r.get("duration_months", 0) for r in completed_roles) / len(completed_roles)
        # 24 months (2 years) is the ideal minimum per role for this JD
        tenure_score = _clamp(avg_tenure / 24)
    else:
        tenure_score = 0.8  # Only one role, probably current — no penalty

    # --- Title relevance ---
    current_title = _normalise(profile.get("current_title", ""))

    strong_title_match = any(keyword in current_title for keyword in STRONG_TITLE_KEYWORDS)
    weak_title_match = any(keyword in current_title for keyword in WEAK_TITLE_KEYWORDS)

    if strong_title_match:
        title_score = 1.0
    elif weak_title_match:
        title_score = 0.1
    else:
        title_score = 0.5  # Neutral / unknown title

    # --- Production signals in descriptions ---
    all_descriptions = " ".join(
        _normalise(role.get("description", "")) for role in career_history
    )
    production_count = sum(
        1 for keyword in PRODUCTION_KEYWORDS if keyword in all_descriptions
    )
    production_score = _clamp(production_count / 6)  # 6+ production keywords = full score

    # Combine sub-scores
    raw_score = (
        0.30 * product_score +
        0.20 * tenure_score +
        0.25 * title_score +
        0.25 * production_score
    )

    # Apply consulting penalty as a multiplier
    return _clamp(raw_score * consulting_penalty)


# ---------------------------------------------------------------------------
# Behavioral / engagement features
# ---------------------------------------------------------------------------

def behavioral_score(candidate: dict) -> float:
    """Score the candidate's platform engagement and job-seeking behaviour.

    A perfect-on-paper candidate who hasn't logged in for 6 months and has
    a 5% response rate is — for hiring purposes — not actually available.

    Returns:
        Float in [0, 1].
    """
    signals = candidate.get("redrob_signals", {})

    # --- Recency: when did they last log in? ---
    days_since_active = _days_since(signals.get("last_active_date"))
    if days_since_active is None:
        recency_score = 0.4
    elif days_since_active <= 7:
        recency_score = 1.0
    elif days_since_active <= 30:
        recency_score = 0.85
    elif days_since_active <= STALENESS_CUTOFF_DAYS:
        recency_score = 0.5
    else:
        recency_score = 0.1  # Inactive for 3+ months → likely not looking

    # --- Open to work flag ---
    open_to_work = signals.get("open_to_work_flag", False)
    open_score = 1.0 if open_to_work else 0.5

    # --- Recruiter response rate ---
    response_rate = signals.get("recruiter_response_rate", 0.0)
    response_score = _clamp(response_rate)

    # --- Response speed (lower avg_response_time = better) ---
    avg_hours = signals.get("avg_response_time_hours", 168)  # default 1 week
    # 2 hours = excellent, 24 hours = good, 168 hours (1 week) = poor
    speed_score = _clamp(1.0 - (avg_hours / 168))

    # --- Interview completion rate (shows reliability) ---
    interview_rate = signals.get("interview_completion_rate", 0.5)
    interview_score = _clamp(interview_rate)

    # --- Offer acceptance rate (shows genuine interest) ---
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate == -1:
        offer_score = 0.6  # No history — neutral, not negative
    else:
        offer_score = _clamp(offer_rate)

    # --- Active job seeking (applied to roles recently) ---
    applications = signals.get("applications_submitted_30d", 0)
    active_seeker_score = 1.0 if applications > 0 else 0.5

    # --- Recruiter demand signal (others are interested too) ---
    saved_count = signals.get("saved_by_recruiters_30d", 0)
    demand_score = _clamp(min(saved_count, 10) / 10)  # cap at 10 saves

    # Combine sub-scores with weights
    score = (
        0.20 * recency_score +
        0.15 * open_score +
        0.20 * response_score +
        0.10 * speed_score +
        0.15 * interview_score +
        0.10 * offer_score +
        0.05 * active_seeker_score +
        0.05 * demand_score
    )

    return _clamp(score)


# ---------------------------------------------------------------------------
# Experience fit features
# ---------------------------------------------------------------------------

def experience_fit_score(candidate: dict) -> float:
    """Score how well the candidate's experience level and depth matches the JD.

    The JD targets 5-9 years with hands-on production ML/AI work.

    Returns:
        Float in [0, 1].
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career_history = candidate.get("career_history", [])

    # --- Years of experience band ---
    yoe = profile.get("years_of_experience", 0)

    if IDEAL_YOE_MIN <= yoe <= IDEAL_YOE_MAX:
        yoe_score = 1.0
    elif yoe < IDEAL_YOE_MIN:
        # Under-experienced: 4 years gets 0.6, 3 years gets 0.4, etc.
        yoe_score = _clamp(0.5 + (yoe / IDEAL_YOE_MIN) * 0.5)
    else:
        # Over-experienced is less penalised (10 years is fine; 20 years less so)
        excess = yoe - IDEAL_YOE_MAX
        yoe_score = _clamp(1.0 - (excess / 20) * 0.4)

    # --- GitHub activity (proxy for active coding) ---
    github_score_raw = signals.get("github_activity_score", -1)
    if github_score_raw == -1:
        # No GitHub linked — neutral, not a hard penalty for senior folks
        github_score = 0.4
    else:
        github_score = _clamp(github_score_raw / 100)

    # --- AI/ML title history ---
    ai_title_months = 0
    total_months = 0
    ai_title_keywords = ["ml", "machine learning", "ai", "nlp", "data scientist", "research", "ranking", "search"]

    for role in career_history:
        duration = role.get("duration_months", 0)
        title_lower = _normalise(role.get("title", ""))
        total_months += duration

        if any(kw in title_lower for kw in ai_title_keywords):
            ai_title_months += duration

    ai_ratio = ai_title_months / max(total_months, 1)
    ai_history_score = _clamp(ai_ratio * 1.5)  # reward heavy AI background

    # Combine
    score = (
        0.40 * yoe_score +
        0.30 * github_score +
        0.30 * ai_history_score
    )

    return _clamp(score)


# ---------------------------------------------------------------------------
# Logistics features
# ---------------------------------------------------------------------------

def logistics_score(candidate: dict) -> float:
    """Score logistical fit: location, notice period, salary, work mode.

    Returns:
        Float in [0, 1].
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    # --- Location ---
    location = _normalise(profile.get("location", ""))
    country = _normalise(profile.get("country", ""))
    willing_to_relocate = signals.get("willing_to_relocate", False)

    location_matched = any(city in location for city in PREFERRED_LOCATIONS)

    if location_matched and country == "india":
        location_score = 1.0
    elif country == "india" and willing_to_relocate:
        location_score = 0.8
    elif country == "india":
        location_score = 0.6
    elif willing_to_relocate:
        location_score = 0.4  # Outside India but willing to relocate
    else:
        location_score = 0.2

    # --- Notice period ---
    notice_days = signals.get("notice_period_days", 90)

    if notice_days <= NOTICE_IDEAL_DAYS:
        notice_score = 1.0
    elif notice_days <= NOTICE_OK_DAYS:
        notice_score = 0.7
    elif notice_days <= NOTICE_ACCEPTABLE_DAYS:
        notice_score = 0.5
    else:
        notice_score = 0.3  # 90+ days notice → longer wait

    # --- Salary alignment ---
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    salary_min = salary_range.get("min", 0)
    salary_max = salary_range.get("max", 0)

    if salary_min == 0 and salary_max == 0:
        salary_score = 0.7  # Not specified — neutral
    elif salary_min > SALARY_BAND_MAX_LPA:
        salary_score = 0.2  # Way above budget
    elif salary_max < SALARY_BAND_MIN_LPA * 0.5:
        salary_score = 0.5  # Below band (might not be right seniority)
    else:
        salary_score = 1.0  # Overlaps with expected band

    # --- Work mode preference (JD is hybrid) ---
    preferred_mode = signals.get("preferred_work_mode", "flexible")
    work_mode_scores = {
        "hybrid":   1.0,
        "flexible": 0.9,
        "onsite":   0.7,
        "remote":   0.5,  # JD says hybrid, so remote is less ideal
    }
    work_mode_score = work_mode_scores.get(preferred_mode, 0.7)

    # Combine
    score = (
        0.40 * location_score +
        0.30 * notice_score +
        0.20 * salary_score +
        0.10 * work_mode_score
    )

    return _clamp(score)


# ---------------------------------------------------------------------------
# Summary extractor (used by reasoning.py)
# ---------------------------------------------------------------------------

def extract_key_facts(candidate: dict) -> dict:
    """Extract human-readable facts for use in reasoning strings.

    Returns a dict of named facts that reasoning.py can reference directly.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career_history = candidate.get("career_history", [])

    # Find the most recent non-consulting role
    product_roles = [
        r for r in career_history if not _is_consulting_firm(r.get("company", ""))
    ]
    best_role = product_roles[0] if product_roles else (career_history[0] if career_history else {})

    days_since_active = _days_since(signals.get("last_active_date"))

    return {
        "candidate_id":       candidate.get("candidate_id", ""),
        "years_exp":          profile.get("years_of_experience", 0),
        "current_title":      profile.get("current_title", "Unknown"),
        "current_company":    profile.get("current_company", "Unknown"),
        "location":           profile.get("location", "Unknown"),
        "best_company":       best_role.get("company", profile.get("current_company", "")),
        "best_title":         best_role.get("title", profile.get("current_title", "")),
        "notice_days":        signals.get("notice_period_days", 90),
        "response_rate":      signals.get("recruiter_response_rate", 0),
        "github_score":       signals.get("github_activity_score", -1),
        "open_to_work":       signals.get("open_to_work_flag", False),
        "days_since_active":  days_since_active,
        "top_skills":         [s["name"] for s in candidate.get("skills", [])[:4]],
        "interview_rate":     signals.get("interview_completion_rate", 0),
        "notice_period_days": signals.get("notice_period_days", 90),
    }
