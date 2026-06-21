"""
precompute.py — Offline embedding generation (run once before ranking).

This script reads all 100,000 candidates and computes MiniLM semantic
similarity scores against the job description. Results are saved to the
artifacts/ folder and loaded during the fast ranking step.

Usage:
    python precompute.py --candidates ./data/candidates.jsonl

Output:
    artifacts/semantic_scores.npy    — float32 array of shape (N,)
    artifacts/candidate_ids.json     — ordered list of candidate IDs
    artifacts/id_to_index.json       — mapping of candidate_id → array index

Runtime: ~3–8 minutes on CPU for 100K candidates with MiniLM-L6-v2.
Memory:  ~2–4 GB peak.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
BATCH_SIZE = 512  # Number of candidates to encode at once — tune for your RAM


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-compute MiniLM semantic similarity scores for all candidates."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidates.jsonl",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_miniLM_model():
    """Load the MiniLM sentence-transformers model.

    Uses 'all-MiniLM-L6-v2' — fast, ~80 MB, excellent for semantic similarity.
    Model is downloaded once and cached by sentence-transformers automatically.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    logger.info("Loading MiniLM model (downloads ~80 MB on first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model loaded.")
    return model


def embed_jd(model, jd_text: str) -> np.ndarray:
    """Embed the job description text into a vector.

    Args:
        model:   The loaded SentenceTransformer model.
        jd_text: The full JD text string from config.py.

    Returns:
        A 1D numpy array of shape (384,) — the JD embedding.
    """
    logger.info("Embedding job description...")
    jd_vector = model.encode(jd_text, convert_to_numpy=True, normalize_embeddings=True)
    logger.info("JD embedding shape: %s", jd_vector.shape)
    return jd_vector


def build_candidate_text(candidate: dict) -> str:
    """Build a single text string representing a candidate's full profile.

    Combines: headline + summary + career descriptions + skill names.
    The richer the input text, the better the semantic match.
    Optimised for CPU speed by truncating long descriptions.
    """
    parts: list[str] = []

    profile = candidate.get("profile", {})

    if profile.get("headline"):
        parts.append(profile["headline"])

    if profile.get("summary"):
        parts.append(profile["summary"][:200])  # limit summary length

    # Focus on titles and first 150 chars of descriptions
    for role in candidate.get("career_history", []):
        if role.get("title"):
            parts.append(role["title"])
        if role.get("description"):
            parts.append(role["description"][:150])

    skill_names = [s["name"] for s in candidate.get("skills", []) if s.get("name")]
    if skill_names:
        parts.append(", ".join(skill_names[:15]))  # limit skills to top 15

    cert_names = [c["name"] for c in candidate.get("certifications", []) if c.get("name")]
    if cert_names:
        parts.append(", ".join(cert_names[:5]))

    # Join and limit total characters to 1000 to keep Transformer sequence short
    full_text = " ".join(parts)
    return full_text[:1000]


def stream_candidates(jsonl_path: Path):
    """Yield one candidate dict at a time from the JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning("Skipping bad JSON on line %d: %s", line_number, error)


def cosine_similarity_batch(candidate_matrix: np.ndarray, jd_vector: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a batch of candidate vectors and the JD vector.

    Both inputs should already be L2-normalised (sentence-transformers does this
    when normalize_embeddings=True), so cosine similarity = dot product.

    Args:
        candidate_matrix: Shape (batch_size, 384) — normalised embeddings.
        jd_vector:        Shape (384,) — normalised JD embedding.

    Returns:
        Shape (batch_size,) — similarity scores in [-1, 1], clipped to [0, 1].
    """
    similarities = candidate_matrix @ jd_vector
    return np.clip(similarities, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main pre-computation pipeline
# ---------------------------------------------------------------------------

def precompute(candidates_path: Path) -> None:
    """Run the full pre-computation pipeline and save artifacts.

    Steps:
    1. Load MiniLM model
    2. Embed the JD
    3. Stream all candidates, batch-encode their text
    4. Compute cosine similarity for each batch
    5. Save scores and candidate ID mappings to artifacts/

    Args:
        candidates_path: Path to candidates.jsonl
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_miniLM_model()

    # Import JD text from config
    from ranker.config import JD_TEXT
    jd_vector = embed_jd(model, JD_TEXT)

    # --- Stream and encode candidates in batches ---
    candidate_ids: list[str] = []
    all_scores: list[float] = []

    batch_texts: list[str] = []
    batch_ids: list[str] = []

    logger.info("Streaming and encoding candidates from %s ...", candidates_path)
    start_time = time.time()

    with tqdm(desc="Encoding", unit="candidates") as progress_bar:
        for candidate in stream_candidates(candidates_path):
            candidate_id = candidate.get("candidate_id", "")
            text = build_candidate_text(candidate)

            batch_ids.append(candidate_id)
            batch_texts.append(text)

            if len(batch_texts) >= BATCH_SIZE:
                # Encode this batch
                batch_matrix = model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                scores = cosine_similarity_batch(batch_matrix, jd_vector)

                candidate_ids.extend(batch_ids)
                all_scores.extend(scores.tolist())

                progress_bar.update(len(batch_texts))
                batch_texts = []
                batch_ids = []

        # Encode any remaining candidates in the final partial batch
        if batch_texts:
            batch_matrix = model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores = cosine_similarity_batch(batch_matrix, jd_vector)
            candidate_ids.extend(batch_ids)
            all_scores.extend(scores.tolist())
            progress_bar.update(len(batch_texts))

    elapsed = time.time() - start_time
    logger.info("Encoded %d candidates in %.1f seconds.", len(candidate_ids), elapsed)

    # --- Save artifacts ---
    scores_array = np.array(all_scores, dtype=np.float32)
    scores_path = ARTIFACTS_DIR / "semantic_scores.npy"
    np.save(str(scores_path), scores_array)
    logger.info("Saved semantic scores → %s  (shape: %s)", scores_path, scores_array.shape)

    ids_path = ARTIFACTS_DIR / "candidate_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)
    logger.info("Saved candidate IDs → %s", ids_path)

    id_to_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    index_path = ARTIFACTS_DIR / "id_to_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(id_to_index, f)
    logger.info("Saved ID-to-index mapping → %s", index_path)

    logger.info("Pre-computation complete. Run rank.py to produce the submission CSV.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not args.candidates.exists():
        logger.error("candidates.jsonl not found at: %s", args.candidates)
        sys.exit(1)

    precompute(args.candidates)
