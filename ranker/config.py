"""
config.py — All constants, JD configuration, and scoring weights.

Everything that might need tuning lives here.
No magic numbers scattered across other files.
"""

# ---------------------------------------------------------------------------
# Job Description Configuration
# ---------------------------------------------------------------------------

# The full JD text used for MiniLM semantic embedding.
# This is what every candidate's text gets compared against.
JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI Series A talent intelligence platform Pune Noida India Hybrid.

Required skills: production embeddings retrieval systems sentence-transformers BGE E5 OpenAI embeddings
deployed real users embedding drift index refresh retrieval quality regression production.
Vector databases hybrid search infrastructure Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS
operational experience production deployment.
Strong Python code quality.
Evaluation frameworks ranking systems NDCG MRR MAP offline online A/B test correlation ranking evaluation.

Nice to have: LLM fine-tuning LoRA QLoRA PEFT learning to rank XGBoost neural ranking models
HR tech recruiting marketplace distributed systems large scale inference open source contributions AI ML.

Ideal candidate: 6 to 8 years applied ML AI product companies shipped end to end ranking search
recommendation systems real users meaningful scale hybrid retrieval dense retrieval LLM integration
re-ranking semantic search vector search production deployment recruiter engagement metrics
evaluation infrastructure A/B testing feedback loops mentoring team building.

NOT wanted: pure research academic no production deployment consulting firms entire career TCS Infosys Wipro
Accenture Cognizant Capgemini title chasing framework enthusiast LangChain tutorial only
computer vision speech robotics without NLP IR closed source only no external validation.
"""

# Core skills the JD explicitly requires — strong positive signal
JD_REQUIRED_SKILLS: list[str] = [
    "embeddings",
    "retrieval",
    "vector database",
    "vector db",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "faiss",
    "elasticsearch",
    "opensearch",
    "sentence-transformers",
    "bge",
    "e5",
    "hybrid search",
    "dense retrieval",
    "semantic search",
    "ranking",
    "python",
    "ndcg",
    "mrr",
    "evaluation framework",
    "nlp",
    "information retrieval",
    "re-ranking",
    "recommendation",
    "search",
]

# Nice-to-have skills — moderate positive signal
JD_NICE_TO_HAVE_SKILLS: list[str] = [
    "lora",
    "qlora",
    "peft",
    "fine-tuning",
    "fine tuning",
    "xgboost",
    "learning to rank",
    "distributed systems",
    "open source",
    "hr tech",
    "rag",
    "llm",
    "transformers",
    "hugging face",
    "mlops",
    "a/b testing",
    "feature store",
    "kubeflow",
]

# Roles that are clearly a strong match for "Senior AI Engineer"
STRONG_TITLE_KEYWORDS: list[str] = [
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "nlp engineer",
    "data scientist",
    "research engineer",
    "applied scientist",
    "search engineer",
    "ranking engineer",
    "recommendation engineer",
    "retrieval engineer",
]

# Roles that suggest the candidate is non-technical for this JD
WEAK_TITLE_KEYWORDS: list[str] = [
    "hr manager",
    "marketing manager",
    "content writer",
    "graphic designer",
    "accountant",
    "civil engineer",
    "mechanical engineer",
    "sales executive",
    "customer support",
    "operations manager",
    "business analyst",
    "project manager",
]

# Production signals — words in career descriptions that indicate real deployment
PRODUCTION_KEYWORDS: list[str] = [
    "production",
    "deployed",
    "shipped",
    "users",
    "scale",
    "serving",
    "latency",
    "throughput",
    "real-time",
    "online",
    "a/b test",
    "experiment",
    "inference",
    "api",
]

# Companies considered big IT consulting/services firms.
# Penalise only if the candidate's ENTIRE career is here.
CONSULTING_FIRMS: set[str] = {
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "mphasis",
    "hexaware",
    "tech mahindra",
    "hcl",
    "l&t infotech",
    "ltimindtree",
    "ibm",
    "dxc technology",
    "unisys",
}

# Preferred locations for this role (Pune/Noida + nearby metros)
PREFERRED_LOCATIONS: set[str] = {
    "pune",
    "noida",
    "delhi",
    "new delhi",
    "gurgaon",
    "gurugram",
    "hyderabad",
    "mumbai",
    "bengaluru",
    "bangalore",
    "ncr",
}

# ---------------------------------------------------------------------------
# Scoring Weights  (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "semantic":    0.35,   # MiniLM cosine similarity vs JD
    "career":      0.25,   # Product-company exp, tenure, disqualifiers
    "behavioral":  0.20,   # Recency, response rate, availability
    "experience":  0.15,   # Years band, production signals, GitHub
    "logistics":   0.05,   # Location, notice, salary, work mode
}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Experience band the JD targets
IDEAL_YOE_MIN: int = 5
IDEAL_YOE_MAX: int = 9

# Notice period thresholds (days)
NOTICE_IDEAL_DAYS: int = 30
NOTICE_OK_DAYS: int = 60
NOTICE_ACCEPTABLE_DAYS: int = 90

# How many days since last login before we consider candidate "stale"
STALENESS_CUTOFF_DAYS: int = 90

# Minimum recruiter response rate to be considered "responsive"
MIN_RESPONSE_RATE: float = 0.3

# Salary band for Senior AI Engineer in India (INR LPA)
SALARY_BAND_MIN_LPA: float = 25.0
SALARY_BAND_MAX_LPA: float = 80.0
