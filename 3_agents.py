# agents.py
import importlib.util
from typing import List, Tuple
from langchain_core.documents import Document

# Import numbered modules using importlib
spec_groq = importlib.util.spec_from_file_location("groq_models", "1_groq_models.py")
groq_models = importlib.util.module_from_spec(spec_groq)
spec_groq.loader.exec_module(groq_models)

spec_rag = importlib.util.spec_from_file_location("rag_backend", "2_rag_backend.py")
rag_backend = importlib.util.module_from_spec(spec_rag)
spec_rag.loader.exec_module(rag_backend)

groq_chat = groq_models.groq_chat
ROUTER_MODEL = groq_models.ROUTER_MODEL
ANSWER_MODEL = groq_models.ANSWER_MODEL
GRADER_MODEL = groq_models.GRADER_MODEL
GUARD_MODEL = groq_models.GUARD_MODEL
retrieve_docs = rag_backend.retrieve_docs
web_search_context = rag_backend.web_search_context

# Router and rewriter
def router_agent(question: str) -> str:
    """
    Decide: DIRECT, TOOL, REWRITE, or MULTIHOP.
    """
    question_lower = question.lower()
    
    # Quick heuristics for common patterns
    multi_hop_keywords = ["then", "first...then", "compare", "difference between", "and also"]
    tool_keywords = ["how to", "how do", "what is", "explain", "documentation", "tutorial", 
                     "recipe", "cookbook", "guide", "steps", "recommend", "hugging face docs"]
    
    # Check for multi-hop patterns
    if any(keyword in question_lower for keyword in multi_hop_keywords):
        if "then" in question_lower or "compare" in question_lower:
            return "MULTIHOP"
    
    # Check if question asks about specific documentation/procedures
    if any(keyword in question_lower for keyword in ["hugging face", "hf docs", "documentation", 
                                                     "cookbook", "tutorial", "how to"]):
        return "TOOL"
    
    prompt = f"""You are a routing controller for an advanced agentic RAG system over Hugging Face documentation.

Decide how to handle the question:

- DIRECT  : Simple definitional questions answerable without retrieval (e.g., "What is X?" for basic concepts).
- TOOL    : Questions requiring Hugging Face documentation, tutorials, recipes, or specific implementation details.
- REWRITE : Questions that need query reformulation for better retrieval.
- MULTIHOP: Questions requiring multiple sequential sub-questions (e.g., "First X, then Y" or "Compare X and Y").

Examples:
- "What is RAG?" → DIRECT (basic definition)
- "How do I load a dataset?" → TOOL (needs documentation)
- "What does HF docs say about X?" → TOOL (explicitly asks for docs)
- "First explain X, then explain Y" → MULTIHOP (sequential sub-questions)
- "Compare X and Y" → MULTIHOP (needs decomposition)

Return ONLY ONE WORD: DIRECT, TOOL, REWRITE, or MULTIHOP.

Question: {question}
"""
    out = groq_chat(
        ROUTER_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=8,
    ).strip().upper()

    if "MULTIHOP" in out:
        return "MULTIHOP"
    if "TOOL" in out:
        return "TOOL"
    if "REWRITE" in out:
        return "REWRITE"
    # Default to TOOL for ambiguous cases about documentation
    if any(kw in question_lower for kw in ["docs", "documentation", "hugging face", "hf"]):
        return "TOOL"
    return "DIRECT"


def rewrite_agent(question: str) -> str:
    prompt = f"""
Rewrite the user's question to be clearer and more specific for retrieval over documentation.
Do NOT answer.

Original question: {question}
"""
    return groq_chat(
        ROUTER_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.3,
    ).strip()

# Multi‑hop decomposition agent
def decompose_agent(question: str) -> List[str]:
    """
    Decompose a complex question into ordered sub-questions.
    LIMITED to 2-3 sub-questions for demo speed optimization.
    """
    prompt = f"""Decompose the following complex question into 2-3 smaller sub-questions
that can be answered sequentially to produce the final answer.

IMPORTANT: Return EXACTLY 2-3 sub-questions maximum. Keep them concise.

Return them as a numbered list, one per line.

Question: {question}"""
    
    out = groq_chat(
        ROUTER_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,  # Limits response length
    )
    subs = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # remove leading "1.", "2)" etc.
        line = line.split(maxsplit=1)[-1] if line[0].isdigit() else line
        subs.append(line)
        # CRITICAL: Limit to 3 sub-questions for speed
        if len(subs) >= 3:
            break
    return subs

# Retrieval grading (CRAG‑style)
def grade_doc_relevance(question: str, doc_text: str) -> str:
    """
    Grade document relevance. Skip grading for very large documents to avoid 413 errors.
    """
    # Skip grading if document is too large
    MAX_DOC_LENGTH = 700
    if len(doc_text) > MAX_DOC_LENGTH:
        # Skip grading on very large docs, assume they're relevant
        return "YES"
    
    MAX_QUESTION_LENGTH = 150
    truncated_question = question[:MAX_QUESTION_LENGTH] if len(question) > MAX_QUESTION_LENGTH else question
    
    prompt = f"""You are grading whether this document is relevant to a user question.

QUESTION:
{truncated_question}

DOCUMENT:
{doc_text[:MAX_DOC_LENGTH]}

If the document helps answer the question, reply with YES.
Otherwise reply with NO.

Reply with ONLY YES or NO."""
    
    try:
        out = groq_chat(
            GRADER_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        ).strip().upper()
        return "YES" if "YES" in out else "NO"
    except Exception:
        # Default to keeping document
        return "YES"


def filter_and_decide(
    question: str,
    docs: List[Document],
    min_keep_ratio: float = 0.5,
    skip_grading: bool = False,  # Skip grading for MULTIHOP speed optimization
) -> Tuple[List[Document], bool]:
    """
    Returns (filtered_docs, need_web_fallback).

    need_web_fallback=True if not enough relevant docs.
    
    For MULTIHOP sub-questions: skip grading to speed up demo.
    """
    if not docs:
        return [], True

    # Skip grading for MULTIHOP sub-questions
    if skip_grading:
        # Just return all docs without grading for speed
        # Also skip web fallback for MULTIHOP to prevent hanging
        return docs, False

    kept: List[Document] = []
    for d in docs:
        if grade_doc_relevance(question, d.page_content) == "YES":
            kept.append(d)

    keep_ratio = len(kept) / max(1, len(docs))
    need_web = keep_ratio < min_keep_ratio
    return kept if kept else docs, need_web

# Answer agent + hallucination / consistency grading
def answer_agent(question: str, context: str | None = None) -> str:
    system = (
        "You are an expert assistant over Hugging Face documentation. "
        "Use CONTEXT when provided. Prefer grounded answers; avoid speculation. "
        "If no context is provided, you can answer from general knowledge about the topic."
    )
    if context and context.strip():
        user = f"Question:\n{question}\n\nCONTEXT:\n{context}\n\nAnswer concisely, explicitly grounded in context."
    else:
        # When no context, still answer but note if it's from general knowledge
        user = f"Question:\n{question}\n\nAnswer concisely based on your knowledge about Hugging Face, RAG, and machine learning."

    return groq_chat(
        ANSWER_MODEL,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=512,
    ).strip()

# Hallucination / quality grading
def grade_answer_against_context(question: str, context: str, answer: str) -> str:
    """
    Return GOOD if answer is grounded & consistent, else BAD (hallucination/low quality).
    """
    # Reduce truncation limits to avoid 413 errors
    MAX_CONTEXT_LENGTH = 1200
    MAX_ANSWER_LENGTH = 600
    MAX_QUESTION_LENGTH = 200
    
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "... [truncated]"
    if len(answer) > MAX_ANSWER_LENGTH:
        answer = answer[:MAX_ANSWER_LENGTH] + "... [truncated]"
    truncated_question = question[:MAX_QUESTION_LENGTH] if len(question) > MAX_QUESTION_LENGTH else question
    
    prompt = f"""You are checking whether an answer is well grounded in the given context.

QUESTION:
{truncated_question}

CONTEXT:
{context}

ANSWER:
{answer}

If the answer is supported by the context and does not contradict it, reply with GOOD.
If the answer uses information not present in the context or contradicts it, reply with BAD.

Reply with ONLY GOOD or BAD."""
    
    try:
        out = groq_chat(
            GRADER_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        ).strip().upper()
        return "GOOD" if "GOOD" in out else "BAD"
    except Exception:
        # Default to GOOD
        return "GOOD"

# Self‑reflection (explanation and potential revision)
def reflection_agent(question: str, context: str, answer: str) -> str:
    """
    Produce a brief self-reflection on answer quality and, if needed, a revised answer.
    """
    # More aggressive truncation for demo stability
    MAX_CONTEXT_LENGTH = 800
    MAX_ANSWER_LENGTH = 500
    MAX_QUESTION_LENGTH = 150
    
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "... [truncated]"
    if len(answer) > MAX_ANSWER_LENGTH:
        answer = answer[:MAX_ANSWER_LENGTH] + "... [truncated]"
    truncated_question = question[:MAX_QUESTION_LENGTH] if len(question) > MAX_QUESTION_LENGTH else question
    
    prompt = f"""You are a self-reflective agent. Keep responses concise and complete.

Given the question, context, and your own answer, first critique the answer
(mention missing points, possible issues), then provide an improved answer if needed.

Format:
CRITIQUE: [2-3 sentences max]
IMPROVED_ANSWER: [1-2 sentences max]

QUESTION:
{truncated_question}

CONTEXT:
{context}

ANSWER:
{answer}"""
    
    try:
        # Increase max_tokens significantly for complete responses
        return groq_chat(
            ANSWER_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        ).strip()
    except Exception:
        improved = answer[:300] + "..." if len(answer) > 300 else answer
        return f"CRITIQUE: Reflection generated with limitations. IMPROVED_ANSWER: {improved}"

# Safety wrapper
def guard_agent(answer: str, question: str) -> str:
    """
    Check if answer is safe. Only block truly harmful content, not legitimate technical questions.
    """
    # Skip guard for obviously safe technical/educational questions
    safe_keywords = ["rag", "retrieval", "dataset", "hugging face", "documentation", 
                     "tutorial", "chunking", "embedding", "vector", "model", "pipeline"]
    question_lower = question.lower()
    if any(keyword in question_lower for keyword in safe_keywords):
        return "OK"
    
    prompt = f"""You are a safety filter for a technical documentation assistant.

Only block answers that contain:
- Illegal content
- Explicitly harmful instructions
- Personal information violations
- Hate speech

Technical and educational content about AI, ML, RAG, and documentation should always be allowed.

QUESTION: {question}
ANSWER: {answer}

Respond with ONLY: OK or BLOCK"""
    try:
        out = groq_chat(
            GUARD_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
        ).strip().upper()
        # Default to OK if we can't parse the response
        return "OK" if "OK" in out or "BLOCK" not in out else "BLOCK"
    except Exception:
        # Default to allowing answer
        return "OK"

def safe_answer_agent(question: str, context: str | None = None) -> str:
    raw = answer_agent(question, context)
    if guard_agent(raw, question) == "BLOCK":
        # For blocked content, explain why instead of generic refusal
        return f"I cannot provide information that violates safety guidelines for the question: {question[:50]}..."
    return raw
