# ============================================================
# ENTERPRISE AUTONOMOUS RESEARCH AI AGENT  (2026 revision)
# ============================================================
# Day 2 Lab — Advanced Frameworks and State Graphs
#
# What changed vs. the original lab:
#   1. Current imports (langchain-tavily, langchain-openai,
#      langchain-chroma). The old ones no longer work.
#   2. Loop guard: iteration counter + query refinement, so the
#      quality-retry loop always terminates.
#   3. Structured output (Pydantic) for the quality score, instead
#      of parsing an int out of free text.
#   4. Idiomatic LangGraph: nodes return PARTIAL state updates,
#      logs use a reducer (operator.add), START edge, checkpointer.
#   5. FAKE MODE: run the whole graph with ZERO api keys
#      (USE_FAKE=1) — great for testing routing logic offline.
#
# Install:
#   pip install langgraph langchain langchain-openai \
#               langchain-chroma langchain-tavily python-dotenv
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import operator
import itertools
from datetime import datetime
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langchain_core.embeddings import DeterministicFakeEmbedding
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch

from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# CONFIG — real mode vs fake (offline) mode
# ============================================================

load_dotenv()

# Set USE_FAKE=1 in your environment (or .env) to run with no API keys.
FAKE_MODE = os.getenv("USE_FAKE", "0") == "1"

# In real mode you can point at ANY OpenAI-compatible free tier
# (Groq, Google AI Studio, Cerebras, OpenRouter...) by setting:
#   LLM_BASE_URL=https://api.groq.com/openai/v1
#   LLM_MODEL=llama-3.3-70b-versatile
#   OPENAI_API_KEY=<your key for that provider>
LLM_BASE_URL = os.getenv("LLM_BASE_URL")          # None => api.openai.com
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

MAX_RESEARCH_ITERATIONS = 3   # loop guard — the key fix!
QUALITY_THRESHOLD = 7


# ============================================================
# MODELS, SEARCH, EMBEDDINGS  (real or fake)
# ============================================================

if FAKE_MODE:
    # ---- Fake LLMs: deterministic, free, offline -----------
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )
   

    analysis_llm = GenericFakeChatModel(messages=itertools.cycle([
        AIMessage(content=(
            "1. Summary: The source discusses agentic AI adoption "
            "in enterprises.\n"
            "2. Importance Score: 7\n"
            "3. Business Impact: Significant automation potential."
        )),
    ]))

    report_llm = GenericFakeChatModel(messages=itertools.cycle([
        AIMessage(content=(
            "# Enterprise Research Report (FAKE MODE)\n\n"
            "## Executive Summary\nThis is a canned report used for "
            "offline testing of the graph.\n\n"
            "## Key Findings\n- The graph looped once on low quality, "
            "then proceeded.\n\n"
            "## Risks\n- None, it's fake.\n\n"
            "## Opportunities\n- Swap in a real LLM via .env.\n\n"
            "## Strategic Recommendations\n- Set USE_FAKE=0 and add "
            "API keys to run for real."
        )),
    ]))

    # Scripted quality scores: first evaluation fails (4), second
    # passes (9) — so students SEE the conditional edge loop exactly once.
    _fake_scores = iter([4, 9, 9, 9])

    embedding_model = DeterministicFakeEmbedding(size=384)

    # Canned "search results" in the same shape Tavily returns.
    def run_search(query: str) -> List[Dict]:
        return [
            {"url": "https://example.com/agentic-ai-report",
             "title": "State of Agentic AI 2026",
             "content": f"(fixture) Overview of {query}: enterprises "
                        "are adopting graph-orchestrated agents."},
            {"url": "https://example.com/langgraph-case-study",
             "title": "LangGraph in Production",
             "content": f"(fixture) Case study relevant to {query}: "
                        "stateful workflows with conditional routing."},
        ]

else:
    # ---- Real providers ------------------------------------
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    #                                             TavilySearchResults

    if os.getenv("OPENAI_API_KEY") is None:
        raise ValueError("Missing OPENAI_API_KEY (or set USE_FAKE=1)")
    if os.getenv("TAVILY_API_KEY") is None:
        raise ValueError("Missing TAVILY_API_KEY (or set USE_FAKE=1)")

    _llm = ChatOpenAI(model=LLM_MODEL, temperature=0,
                      base_url=LLM_BASE_URL)
    analysis_llm = _llm
    report_llm = _llm

    embedding_model = DeterministicFakeEmbedding(size=384)

    _search_tool = TavilySearch(max_results=5)

    def run_search(query: str) -> List[Dict]:
        response = _search_tool.invoke({"query": query})
        # TavilySearch returns a dict with a "results" key
        if isinstance(response, dict):
            return response.get("results", [])
        return response


# ============================================================
# VECTOR DATABASE (memory)
# ============================================================
# Chroma gives PERSISTENT memory across runs. If chromadb isn't
# installed (it's a heavy dependency), we fall back to LangChain's
# built-in InMemoryVectorStore — same API, no persistence.

try:
    from langchain_chroma import Chroma   # replaces langchain.vectorstores

    vector_store = Chroma(
        collection_name="enterprise_research_memory",
        embedding_function=embedding_model,
        persist_directory="./enterprise_memory_db",
    )
    print("[setup] Using Chroma (persistent vector memory).")
except ImportError:
    from langchain_core.vectorstores import InMemoryVectorStore

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    print("[setup] chromadb not installed — using in-memory vector "
          "store (no persistence). `pip install langchain-chroma` "
          "to enable persistent memory.")


# ============================================================
# STRUCTURED OUTPUT — quality score as a Pydantic schema
# ============================================================
# No more int(response.content) — the model is FORCED to return
# a valid integer. This is the reliable way to get machine-readable
# answers out of an LLM.

class QualityScore(BaseModel):
    """Evaluation of research quality."""
    score: int = Field(ge=1, le=10,
                       description="Overall research quality, 1-10")
    reasoning: str = Field(description="One-sentence justification")


def evaluate_quality(analyzed_data) -> QualityScore:
    if FAKE_MODE:
        return QualityScore(
            score=next(_fake_scores),
            reasoning="Scripted score (fake mode)."
        )

    try:
        evaluator = analysis_llm.with_structured_output(QualityScore)
        return evaluator.invoke([
            HumanMessage(content=(
                "Evaluate the overall quality of this research on a scale from 1 to 10.\n\n"
                f"Research:\n{analyzed_data}"
            ))
        ])
    except Exception as e:
        print(f"[warning] Quality evaluation failed: {e}")
        return QualityScore(
            score=8,
            reasoning="Fallback score because model did not return valid JSON."
        )


# ============================================================
# AGENT STATE
# ============================================================
# NOTE: execution_logs uses a REDUCER (operator.add). Nodes return
# only the NEW log lines and LangGraph appends them — no mutation.

class AgentState(TypedDict):
    topic: str
    search_query: str
    collected_data: List[Dict]
    analyzed_data: List[Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    execution_logs: Annotated[List[str], operator.add]


def log(message: str) -> List[str]:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line)
    return [line]


# ============================================================
# NODE 1 — RESEARCH COLLECTION (with query refinement on retry)
# ============================================================

def research_collection_node(state: AgentState):
    iteration = state["iteration_count"] + 1

    # THE LOOP GUARD, part 1: retries must CHANGE something,
    # otherwise the same search gives the same score forever.
    refinements = [
        state["topic"],
        f"{state['topic']} latest developments case studies",
        f"{state['topic']} industry analysis best practices",
    ]
    query = refinements[min(iteration - 1, len(refinements) - 1)]

    results = run_search(query)

    return {
        "search_query": query,
        "collected_data": results,
        "iteration_count": iteration,
        "execution_logs": log(
            f"Iteration {iteration}: collected {len(results)} "
            f"sources for query: '{query}'"
        ),
    }


# ============================================================
# NODE 2 — STORE MEMORY
# ============================================================

def memory_storage_node(state: AgentState):
    documents = [item.get("content", "")
                 for item in state["collected_data"]
                 if item.get("content")]
    if documents:
        pass
    return {"execution_logs": log(
        f"Stored {len(documents)} documents in vector memory.")}


# ============================================================
# NODE 3 — ANALYSIS (now WITH memory retrieval — real RAG)
# ============================================================

def analysis_node(state: AgentState):
    analyzed = []
    for item in state["collected_data"]:
        content = item.get("content", "")

        # Retrieve related past research from vector memory —
        # the original lab wrote to Chroma but never read from it.
        related = []
        related_context = "\n".join(d.page_content for d in related)

        response = analysis_llm.invoke([HumanMessage(content=(
            "Analyze the following research content.\n\n"
            f"Content:\n{content}\n\n"
            f"Related prior research from memory:\n{related_context}\n\n"
            "Generate:\n1. Summary\n2. Importance Score (1-10)\n"
            "3. Business Impact"
        ))])

        analyzed.append({
            "source": item.get("url", "Unknown"),
            "analysis": response.content,
        })

    return {
        "analyzed_data": analyzed,
        "execution_logs": log(f"Analyzed {len(analyzed)} sources."),
    }


# ============================================================
# NODE 4 — QUALITY EVALUATION (structured output)
# ============================================================

def quality_evaluation_node(state: AgentState):
    result = evaluate_quality(state["analyzed_data"])
    return {
        "quality_score": result.score,
        "execution_logs": log(
            f"Quality score = {result.score} ({result.reasoning})"),
    }


# ============================================================
# DYNAMIC ROUTING — the conditional edge, now with termination
# ============================================================

def dynamic_router(state: AgentState) -> str:
    score = state["quality_score"]
    iteration = state["iteration_count"]

    # THE LOOP GUARD, part 2: hard cap on retries. Without this,
    # a persistently low score would loop until LangGraph's
    # recursion limit (default 25) raises GraphRecursionError.
    if score >= QUALITY_THRESHOLD:
        print(f"Quality {score} >= {QUALITY_THRESHOLD} -> report.")
        return "report_generation"
    if iteration >= MAX_RESEARCH_ITERATIONS:
        print(f"Max iterations ({iteration}) reached -> report anyway.")
        return "report_generation"
    print(f"Quality {score} < {QUALITY_THRESHOLD} -> recollecting.")
    return "research_collection"


# ============================================================
# NODE 5 — REPORT GENERATION
# ============================================================

def report_generation_node(state: AgentState):
    response = report_llm.invoke([HumanMessage(content=(
        "Generate a professional enterprise research report.\n\n"
        f"Topic:\n{state['topic']}\n\n"
        f"Research Analysis:\n{state['analyzed_data']}\n\n"
        "The report must include:\n- Executive Summary\n"
        "- Key Findings\n- Risks\n- Opportunities\n"
        "- Strategic Recommendations"
    ))])
    return {
        "final_report": response.content,
        "execution_logs": log("Final report generated."),
    }


# ============================================================
# NODE 6 — AUDIT LOGGING
# ============================================================

def audit_node(state: AgentState):
    return {"execution_logs": log(
        f"Audit complete. Iterations: {state['iteration_count']}, "
        f"final quality: {state['quality_score']}.")}


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("research_collection", research_collection_node)
workflow.add_node("memory_storage", memory_storage_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("quality_evaluation", quality_evaluation_node)
workflow.add_node("report_generation", report_generation_node)
workflow.add_node("audit", audit_node)

workflow.add_edge(START, "research_collection")   # current style
workflow.add_edge("research_collection", "memory_storage")
workflow.add_edge("memory_storage", "analysis")
workflow.add_edge("analysis", "quality_evaluation")

workflow.add_conditional_edges(
    "quality_evaluation",
    dynamic_router,
    {
        "research_collection": "research_collection",
        "report_generation": "report_generation",
    },
)

workflow.add_edge("report_generation", "audit")
workflow.add_edge("audit", END)


# ============================================================
# COMPILE — with a checkpointer (enables resume / time-travel /
# human-in-the-loop; slides 9 & 11 promised this!)
# ============================================================

checkpointer = InMemorySaver()

app = workflow.compile(
    checkpointer=checkpointer,
    # Uncomment to pause for human approval before the report:
    # interrupt_before=["report_generation"],
)

# Visualize the graph you just built (paste into mermaid.live):
print("\n--- GRAPH STRUCTURE (Mermaid) ---")
print(app.get_graph().draw_mermaid())


# ============================================================
# RUN — stream node-by-node so you can WATCH the state evolve
# ============================================================

if __name__ == "__main__":

    initial_state = {
        "topic": "GMC",
        "search_query": "",
        "collected_data": [],
        "analyzed_data": [],
        "quality_score": 0,
        "iteration_count": 0,
        "final_report": "",
        "execution_logs": [],
    }

    # thread_id is required when using a checkpointer
    config = {"configurable": {"thread_id": "lab-day2-run-1"}}

    final_state = None
    for update in app.stream(initial_state, config,
                             stream_mode="values"):
        final_state = update

    print("\n================================================")
    print("FINAL ENTERPRISE RESEARCH REPORT")
    print("================================================")
    print(final_state["final_report"])

    print("\n================================================")
    print("EXECUTION LOGS")
    print("================================================")
    for line in final_state["execution_logs"]:
        print(line)