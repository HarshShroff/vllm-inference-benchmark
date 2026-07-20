"""
Fixed prompt set for reproducible benchmarking across engines.

Prompts are grouped by approximate input length (short / medium / long) to
measure how serving performance scales with context length, not just batch size.
Using a fixed set (rather than random generation) makes runs comparable across
vLLM and the naive baseline, and across repeated runs over time.
"""

SHORT_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in two sentences.",
    "What is the time complexity of binary search?",
    "Summarize the plot of a mystery novel in one paragraph.",
    "Write a haiku about distributed systems.",
    "What are the main causes of the 2008 financial crisis?",
]

MEDIUM_PROMPTS = [
    """You are reviewing a pull request for a production RAG pipeline. The PR adds a new
    chunking strategy that splits documents by semantic similarity rather than fixed token
    windows. Write a code review comment identifying two potential risks with this approach
    and one suggestion for mitigating each risk.""",
    """A user reports that their computer vision model, deployed on an NVIDIA Jetson Orin,
    is running at 15 FPS instead of the expected 50 FPS after a recent TensorRT engine
    rebuild. List five likely causes, ordered by probability, and a diagnostic step for each.""",
    """Explain how PagedAttention reduces GPU memory fragmentation compared to naive KV-cache
    allocation in transformer inference serving. Include a brief analogy to OS virtual memory
    paging.""",
]

LONG_PROMPTS = [
    """You are architecting a multi-agent research system. The system has 11 specialized
    agents: a query planner, five domain-specific researchers, a fact-checker, a citation
    verifier, a synthesis agent, an LLM-as-Judge evaluator, and a final formatter. Agents
    communicate through a shared state graph implemented in LangGraph. Design a memory
    architecture that prevents agents from reinforcing each other's factual errors while
    still allowing them to build on each other's partial findings. Discuss: (1) what should
    be agent-local vs. shared state, (2) how you would detect and interrupt an error-reinforcement
    loop before it propagates to the final synthesis, and (3) how the LLM-as-Judge evaluator's
    feedback should be routed back into the pipeline -- as a hard rejection, a soft re-ranking
    signal, or a targeted re-query to specific agents. Be specific about tradeoffs.""",
    """Write a detailed incident postmortem for the following scenario: A production RAG
    system serving 2,500+ regulatory documents began returning degraded retrieval quality
    over a 6-hour window. P99 latency increased from 800ms to 4.2s, and a spot-check of
    20 queries showed 6 with clearly irrelevant retrieved chunks. The vector database is
    pgvector on RDS, embeddings are generated via a hosted API, and the deployment uses
    a blue-green rollout strategy. Walk through a plausible root cause investigation,
    identify the most likely root cause, and propose both an immediate mitigation and a
    longer-term structural fix to prevent recurrence.""",
]

ALL_PROMPTS = {
    "short": SHORT_PROMPTS,
    "medium": MEDIUM_PROMPTS,
    "long": LONG_PROMPTS,
}


def get_prompt_batch(size: int, length_bucket: str = "medium") -> list[str]:
    """
    Return `size` prompts from the given length bucket, cycling through the
    fixed set if size exceeds the number of unique prompts available.
    """
    pool = ALL_PROMPTS[length_bucket]
    return [pool[i % len(pool)] for i in range(size)]
