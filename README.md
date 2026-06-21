# Redrob Intelligent Candidate Ranker

A semantic + behavioural AI recruiting system for the [Redrob Intelligent Candidate Discovery & Ranking Challenge](https://redrob.ai).

Ranks 100,000 candidates for a **Senior AI Engineer** role using:
- **MiniLM-L6-v2** semantic similarity against the Job Description
- **Multi-signal structured scoring** (career quality, behavioural engagement, logistics)
- **Honeypot detection** to eliminate fraudulent profiles

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the candidates file
```
data/candidates.jsonl      ← copy the 487 MB file here
```

### 3. Pre-compute MiniLM embeddings (run once — ~3–8 min)
```bash
python precompute.py --candidates ./data/candidates.jsonl
```
This generates:
```
artifacts/semantic_scores.npy
artifacts/candidate_ids.json
artifacts/id_to_index.json
```

### 4. Rank and produce output files (≤5 min)
```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```
This generates:
```
submission.csv      ← required competition format
submission.xlsx     ← formatted Excel with colour-coded ranks
```

### 5. Validate before submitting
```bash
python validate_submission.py submission.csv
```

---

## Output Files

### `submission.csv`
Competition-required format:
```
candidate_id,rank,score,reasoning
CAND_0042871,1,0.9187,"ML Engineer at Flipkart (6 yrs); key skills: embeddings, FAISS; 14-day notice."
CAND_0019884,2,0.9043,"Applied ML background at Swiggy + Razorpay; key skills: retrieval, Python; 30-day notice."
...
```

### `submission.xlsx`
Same data in Excel with:
- 🟦 Bold blue header row
- 🟩 Green background for Rank 1–10
- 🟨 Yellow background for Rank 11–50
- ⬜ White background for Rank 51–100
- Extra columns: Title, Company, Location, Years of Experience

---

## Architecture

```
Score = 0.35 × SemanticScore (MiniLM cosine similarity vs JD)
      + 0.25 × CareerScore   (product-company exp, tenure, titles, production signals)
      + 0.20 × BehaviouralScore (recency, response rate, availability, engagement)
      + 0.15 × ExperienceScore  (years band, GitHub activity, AI role history)
      + 0.05 × LogisticsScore   (location, notice period, salary, work mode)
      × HoneypotMultiplier      (0.0 if detected, 1.0 otherwise)
```

### Key Design Decisions

1. **MiniLM over pure keyword matching** — catches candidates who describe
   "built a recommendation system at scale" without saying "vector database"
2. **Consulting-firm penalty** — only applied if 95%+ of career is at services firms
3. **Behavioural signals as a multiplier** — a perfect-on-paper candidate who
   hasn't logged in for 3 months is down-weighted
4. **Honeypot detection** — 6 checks: salary anomaly, impossible skill durations,
   expert-with-zero-months, timeline paradox, ghost profile, excessive expert claims

---

## Project Structure

```
redrob-ranker/
├── rank.py              ← 🎯 Main entry point (≤5 min ranking step)
├── precompute.py        ← Offline embedding generation (run once)
├── requirements.txt
├── .gitignore
│
├── ranker/
│   ├── config.py        ← All constants and JD configuration
│   ├── loader.py        ← JSONL streaming
│   ├── features.py      ← Feature extraction (5 categories)
│   ├── honeypot.py      ← Honeypot detection (6 checks)
│   ├── scorer.py        ← Composite score assembly
│   └── reasoning.py     ← Per-candidate reasoning text
│
├── data/                ← Place candidates.jsonl here (gitignored)
└── artifacts/           ← Pre-computed scores (gitignored)
```

---

## Compute Environment

| Constraint | Value |
|-----------|-------|
| Runtime (ranking step) | ≤ 5 minutes |
| RAM | ≤ 16 GB |
| Compute | CPU only |
| Network during ranking | None |

Tested on: Windows 11, Python 3.12, 16 GB RAM

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | MiniLM-L6-v2 semantic embeddings |
| `numpy` | Fast array operations |
| `scikit-learn` | Cosine similarity utilities |
| `openpyxl` | Excel (.xlsx) output |
| `tqdm` | Progress bars |
