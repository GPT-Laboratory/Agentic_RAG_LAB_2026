# graph_workflow.py
import importlib.util
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

# Import numbered modules using importlib
spec_agents = importlib.util.spec_from_file_location("agents", "3_agents.py")
agents_module = importlib.util.module_from_spec(spec_agents)
spec_agents.loader.exec_module(agents_module)

spec_rag = importlib.util.spec_from_file_location("rag_backend", "2_rag_backend.py")
rag_backend_module = importlib.util.module_from_spec(spec_rag)
spec_rag.loader.exec_module(rag_backend_module)

# Import from the loaded modules
router_agent = agents_module.router_agent
rewrite_agent = agents_module.rewrite_agent
decompose_agent = agents_module.decompose_agent
filter_and_decide = agents_module.filter_and_decide
safe_answer_agent = agents_module.safe_answer_agent
grade_answer_against_context = agents_module.grade_answer_against_context
reflection_agent = agents_module.reflection_agent
retrieve_docs = rag_backend_module.retrieve_docs
web_search_context = rag_backend_module.web_search_context

# State
class AgentState(TypedDict):
    question: str
    context: Optional[str]
    documents: List[Document]
    sub_questions: List[str]
    current_sub_index: int
    partial_answers: List[str]
    messages: List[BaseMessage]
    rewrite_count: int
    answer_retry_count: int
    used_web_fallback: bool

# Router node and decision
def node_router(state: AgentState) -> AgentState:
    q = state["question"]
    decision = router_agent(q)
    msg = AIMessage(content=f"[ROUTER] {decision}")
    return {**state, "messages": state["messages"] + [msg]}

def route_first(state: AgentState) -> str:
    last = state["messages"][-1].content.upper()
    if "MULTIHOP" in last:
        return "decompose"
    if "TOOL" in last:
        return "retrieve"
    if "REWRITE" in last:
        return "rewrite"
    return "answer"  # DIRECT

# Multi‑hop decomposition node
def node_decompose(state: AgentState) -> AgentState:
    subs = decompose_agent(state["question"])
    msg = AIMessage(content="[DECOMPOSE] " + " | ".join(subs))
    return {
        **state,
        "sub_questions": subs,
        "current_sub_index": 0,
        "partial_answers": [],
        "messages": state["messages"] + [msg],
    }

# Retrieval node with grading + optional web fallback
def node_retrieve(state: AgentState) -> AgentState:
    q = (
        state["sub_questions"][state["current_sub_index"]]
        if state["sub_questions"]
        else state["question"]
    )
    is_multihop = bool(state.get("sub_questions"))
    docs = retrieve_docs(q)
    # Skip grading for MULTIHOP sub-questions
    skip_grading = is_multihop
    filtered_docs, need_web = filter_and_decide(q, docs, min_keep_ratio=0.5, skip_grading=skip_grading)

    ctx = "\n\n".join(d.page_content for d in filtered_docs)
    msg_docs = AIMessage(content=f"[DOCS] kept={len(filtered_docs)} / {len(docs)}")
    msgs = state["messages"] + [msg_docs]

    used_web = state.get("used_web_fallback", False)

    # Skip web search for MULTIHOP sub-questions
    # Only use web search if absolutely necessary and not in MULTIHOP
    if need_web and not used_web and not is_multihop:
        try:
            web_ctx = web_search_context(q)
            ctx = ctx + "\n\n[WEB_FALLBACK]\n" + web_ctx[:500]  # Limit web context
            msgs.append(AIMessage(content="[WEB] used web search fallback"))
            used_web = True
        except Exception:
            # Skip web search if it fails - don't hang
            pass

    msgs.append(AIMessage(content=f"[CONTEXT]\n{ctx[:1500]}"))

    return {
        **state,
        "context": ctx,
        "documents": filtered_docs,
        "used_web_fallback": used_web,
        "messages": msgs,
    }

# Answer + hallucination detection + self‑reflection
MAX_ANSWER_RETRIES = 1

def node_answer(state: AgentState) -> AgentState:
    q = (
        state["sub_questions"][state["current_sub_index"]]
        if state["sub_questions"]
        else state["question"]
    )
    ctx = state.get("context") or ""
    attempt = state.get("answer_retry_count", 0)
    is_multihop = bool(state.get("sub_questions"))

    ans = safe_answer_agent(q, ctx)
    
    # For MULTIHOP sub-questions: skip grading and reflection
    # Only grade/reflect on final synthesized answer for better performance
    if is_multihop:
        # Skip grading and reflection for sub-questions
        msg_ans = AIMessage(content=f"[ANSWER attempt={attempt}] {ans}")
        new_messages = state["messages"] + [msg_ans]
    else:
        # For single-hop: full grading and reflection
        grade = grade_answer_against_context(q, ctx, ans) if ctx else "GOOD"
        refl = reflection_agent(q, ctx, ans)
        msg_ans   = AIMessage(content=f"[ANSWER attempt={attempt}] {ans}")
        msg_grade = AIMessage(content=f"[ANSWER_GRADE] {grade}")
        msg_refl  = AIMessage(content=f"[REFLECTION]\n{refl}")
        new_messages = state["messages"] + [msg_ans, msg_grade, msg_refl]

    # Store partial answers if multi-hop
    partials = list(state.get("partial_answers", []))
    if state["sub_questions"]:
        partials.append(ans)

    return {
        **state,
        "answer_retry_count": attempt + 1,
        "partial_answers": partials,
        "messages": new_messages,
    }

# Routing after answer (retry if BAD):
def route_after_answer(state: AgentState) -> str:
    grades = [m for m in state["messages"] if m.content.startswith("[ANSWER_GRADE]")]
    if not grades:
        return "advance_or_end"
    last = grades[-1].content.upper()
    if "BAD" in last and state["answer_retry_count"] <= MAX_ANSWER_RETRIES:
        return "retrieve"  # try new retrieval (maybe with web fallback)
    return "advance_or_end"

# Multi‑hop advancement or final synthesis: Advance through sub‑questions
def node_advance_or_end(state: AgentState) -> AgentState:
    if not state["sub_questions"]:
        # single-hop: nothing to advance, just return
        return state

    idx = state["current_sub_index"] + 1
    
    # Safety check: prevent infinite loops
    if idx >= len(state["sub_questions"]):
        # Force final synthesis if somehow we're past the end
        combined = "\n\n".join(
            f"Q{i+1}: {q}\nA{i+1}: {a}"
            for i, (q, a) in enumerate(zip(state["sub_questions"], state.get("partial_answers", [])))
        )
        if len(combined) > 2000:
            combined = combined[:2000] + "... [truncated]"
        final = safe_answer_agent(state["question"], context=combined)
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"[FINAL_MULTI_HOP_ANSWER]\n{final}")]
        }
    
    if idx < len(state["sub_questions"]):
        # move to next sub-question
        msg = AIMessage(content=f"[ADVANCE] moving to sub-question {idx}")
        return {
            **state,
            "current_sub_index": idx,
            "context": None,
            "documents": [],
            "answer_retry_count": 0,
            "used_web_fallback": False,
            "messages": state["messages"] + [msg],
        }
    else:
        # all sub-answers collected - synthesize final answer
        combined = "\n\n".join(
            f"Q{i+1}: {q}\nA{i+1}: {a}"
            for i, (q, a) in enumerate(zip(state["sub_questions"], state["partial_answers"]))
        )
        
        # Truncate combined context for final synthesis
        MAX_COMBINED_LENGTH = 2000
        if len(combined) > MAX_COMBINED_LENGTH:
            combined = combined[:MAX_COMBINED_LENGTH] + "... [truncated for synthesis]"
        
        final = safe_answer_agent(
            state["question"],
            context=combined,
        )
        
        # The final answer synthesis is already good enough
        msg_final = AIMessage(content=f"[FINAL_MULTI_HOP_ANSWER]\n{final}")
        
        return {
            **state, 
            "messages": state["messages"] + [msg_final]
        }

# Routing
def route_advance_or_end(state: AgentState) -> str:
    """
    Route after advance_or_end node.
    Safety checks to prevent infinite loops.
    """
    if not state["sub_questions"]:
        return "end"
    
    idx = state["current_sub_index"]
    partials = state.get("partial_answers", [])
    
    # Safety: if we have answers for all sub-questions, we're done
    if len(partials) >= len(state["sub_questions"]):
        return "end"
    
    # Normal routing: continue if more sub-questions to process
    if idx < len(state["sub_questions"]):
        return "retrieve"
    
    # Safety fallback: end if somehow past the end
    return "end"

# Rewriter node and counts
def node_rewrite(state: AgentState) -> AgentState:
    new_q = rewrite_agent(state["question"])
    msg = AIMessage(content=f"[REWRITE] {new_q}")
    return {
        **state,
        "question": new_q,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "messages": state["messages"] + [msg],
    }

# Assemble the graph
workflow = StateGraph(AgentState)

workflow.add_node("router",   node_router)
workflow.add_node("decompose", node_decompose)
workflow.add_node("rewrite",  node_rewrite)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("answer",   node_answer)
workflow.add_node("advance_or_end", node_advance_or_end)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_first,
    {
        "decompose": "decompose",
        "retrieve":  "retrieve",
        "rewrite":   "rewrite",
        "answer":    "answer",
    },
)

workflow.add_edge("decompose", "retrieve")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "answer")  # CRITICAL: TOOL questions need this edge!

workflow.add_conditional_edges(
    "answer",
    route_after_answer,
    {
        "retrieve":        "retrieve",
        "advance_or_end":  "advance_or_end",
    },
)

workflow.add_conditional_edges(
    "advance_or_end",
    route_advance_or_end,
    {
        "retrieve": "retrieve",
        "end":      END,
    },
)

graph = workflow.compile()
