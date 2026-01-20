# groq_models.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Models
ROUTER_MODEL   = "groq/compound-mini"            # routing, rewriting
ANSWER_MODEL   = "openai/gpt-oss-20b"            # main answerer 
GRADER_MODEL   = "groq/compound-mini"            # doc / answer grader (YES/NO or GOOD/BAD decisions)
GUARD_MODEL    = "meta-llama/llama-guard-4-12b"  # safety filter 

def groq_chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """
    Groq chat API call with error handling.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Better error handling - return informative message
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "429" in error_msg:
            return f"[API timeout or rate limit - model: {model}]"
        # Re-raise for other errors to maintain error visibility
        raise
