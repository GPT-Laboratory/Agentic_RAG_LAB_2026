# main.py
import warnings
import numpy as np
import os

# Suppress all warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import importlib.util
from langchain_core.messages import HumanMessage, AIMessage

# Import numbered module using importlib
spec = importlib.util.spec_from_file_location("graph_workflow", "4_graph_workflow.py")
graph_workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_workflow)
graph = graph_workflow.graph
AgentState = graph_workflow.AgentState

def ask_agentic(q: str):
    import time
    start_time = time.time()
    
    state: AgentState = {
        "question": q,
        "context": None,
        "documents": [],
        "sub_questions": [],
        "current_sub_index": 0,
        "partial_answers": [],
        "messages": [HumanMessage(content=q)],
        "rewrite_count": 0,
        "answer_retry_count": 0,
        "used_web_fallback": False,
    }
    
    # Add progress indicator for MULTIHOP questions
    print("🔄 Processing...")
    try:
        result = graph.invoke(state)
        elapsed = time.time() - start_time
        print(f"⏱️  Processing time: {elapsed:.1f}s")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.1f}s: {e}")
        return
    
    # Cleaner output 
    print("\n" + "="*70)
    print(f"❓ QUESTION: {q}")
    print("="*70)
    
    # Extract key information - iterate through messages to find the LAST answers
    router_decision = None
    final_answer = None
    reflection = None
    
    # Process messages to find the latest answers (they may appear multiple times)
    all_answers = []
    all_reflections = []
    
    for m in result["messages"]:
        if isinstance(m, AIMessage):
            content = m.content
            if "[ROUTER]" in content:
                router_decision = content.split("[ROUTER]")[-1].strip()
                print(f"\n🔀 Routing Decision: {router_decision}")
            elif "[FINAL_MULTI_HOP_ANSWER]" in content:
                # Extract multi-hop final answer
                if "\n" in content and "[FINAL_MULTI_HOP_ANSWER]\n" in content:
                    answer_text = content.split("[FINAL_MULTI_HOP_ANSWER]\n", 1)[-1]
                else:
                    answer_text = content.split("[FINAL_MULTI_HOP_ANSWER]", 1)[-1] if "[FINAL_MULTI_HOP_ANSWER]" in content else content
                all_answers.append(("multi-hop", answer_text))
            elif "[ANSWER attempt=" in content:
                # Extract regular answer
                if "] " in content:
                    answer_text = content.split("] ", 1)[1]
                    all_answers.append(("regular", answer_text))
            elif "[REFLECTION]" in content:
                # Extract reflection
                if "[REFLECTION]\n" in content:
                    refl_text = content.split("[REFLECTION]\n", 1)[-1]
                else:
                    refl_text = content.split("[REFLECTION]", 1)[-1] if "[REFLECTION]" in content else content
                all_reflections.append(refl_text)
            elif "[DOCS]" in content:
                print(f"\n📚 {content}")
            elif "[DECOMPOSE]" in content:
                # Show decomposition for multi-hop questions
                sub_questions = content.split("[DECOMPOSE]")[-1].strip()
                print(f"\n🔀 Multi-hop Decomposition: {sub_questions[:150]}..." if len(sub_questions) > 150 else f"\n🔀 Multi-hop Decomposition: {sub_questions}")
            elif "[ADVANCE]" in content:
                print(f"\n⏭️  {content}")
            elif "[WEB]" in content:
                print(f"\n🌐 {content}")
    
    # Use the LAST answer found (prefer multi-hop if available, otherwise use last regular answer)
    if all_answers:
        # Prefer multi-hop answer, otherwise use last regular answer
        multi_hop_answers = [a for t, a in all_answers if t == "multi-hop"]
        if multi_hop_answers:
            final_answer = multi_hop_answers[-1]  # Last multi-hop answer
        else:
            final_answer = all_answers[-1][1]  # Last regular answer
    
    # Use the LAST reflection found
    if all_reflections:
        reflection = all_reflections[-1]
    
    if final_answer:
        print("\n" + "-"*70)
        print("💡 FINAL ANSWER:")
        print("-"*70)
        print(final_answer)
    
    if reflection:
        print("\n" + "-"*70)
        print("💭 SELF-REFLECTION:")
        print("-"*70)
        # Show improved answer if available
        if "IMPROVED_ANSWER:" in reflection:
            parts = reflection.split("IMPROVED_ANSWER:")
            if len(parts) > 1:
                improved = parts[-1].strip()
                print(f"✨ Improved Answer: {improved}")
            else:
                print(reflection[:400] + "..." if len(reflection) > 400 else reflection)
        else:
            print(reflection[:400] + "..." if len(reflection) > 400 else reflection)
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print("Advanced Agentic RAG (Groq, multi-hop, grading, web fallback). Empty line to exit.")
    while True:
        q = input("Q> ").strip()
        if not q:
            break
        ask_agentic(q)
