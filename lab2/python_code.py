# ============================================================
# ENTERPRISE AUTONOMOUS RESEARCH AI AGENT
# ============================================================
# Real Enterprise AI System using:
#
# - LangChain
# - LangGraph
# - OpenAI
# - ChromaDB
# - Tavily Search
# - Stateful AI Workflow
# - Dynamic Graph Orchestration
# - Multi-Step Autonomous Reasoning
#
# ============================================================


# ============================================================
# INSTALL REQUIRED PACKAGES
# ============================================================

# Uncomment if needed

# !pip install langchain
# !pip install langgraph
# !pip install langchain-openai
# !pip install langchain-community
# !pip install chromadb
# !pip install tavily-python
# !pip install python-dotenv


# ============================================================
# IMPORTS
# ============================================================

import os
from typing import TypedDict, List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, END

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from datetime import datetime


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("Missing OPENAI_API_KEY")

if TAVILY_API_KEY is None:
    raise ValueError("Missing TAVILY_API_KEY")


# ============================================================
# INITIALIZE LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# SEARCH TOOL
# ============================================================

search_tool = TavilySearchResults(
    max_results=5
)


# ============================================================
# VECTOR DATABASE
# ============================================================

embedding_model = OpenAIEmbeddings()

vector_store = Chroma(
    collection_name="enterprise_research_memory",
    embedding_function=embedding_model,
    persist_directory="./enterprise_memory_db"
)


# ============================================================
# AGENT STATE
# ============================================================

class AgentState(TypedDict):

    topic: str

    collected_data: List[Dict]

    analyzed_data: List[Dict]

    final_report: str

    execution_logs: List[str]

    quality_score: int


# ============================================================
# LOGGING FUNCTION
# ============================================================

def add_log(state, message):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_message = f"[{timestamp}] {message}"

    print(log_message)

    state["execution_logs"].append(log_message)


# ============================================================
# NODE 1 — RESEARCH COLLECTION
# ============================================================

def research_collection_node(state):

    add_log(state, "Starting autonomous research collection.")

    topic = state["topic"]

    search_results = search_tool.invoke(topic)

    state["collected_data"] = search_results

    add_log(
        state,
        f"Collected {len(search_results)} research sources."
    )

    return state


# ============================================================
# NODE 2 — STORE MEMORY
# ============================================================

def memory_storage_node(state):

    add_log(state, "Storing research into vector database.")

    documents = []

    for item in state["collected_data"]:

        content = item.get("content", "")

        documents.append(content)

    vector_store.add_texts(documents)

    add_log(state, "Research memory updated.")

    return state


# ============================================================
# NODE 3 — ANALYSIS
# ============================================================

def analysis_node(state):

    add_log(state, "Analyzing collected sources.")

    analyzed_results = []

    for item in state["collected_data"]:

        content = item.get("content", "")

        prompt = f"""
        Analyze the following research content.

        Content:
        {content}

        Generate:
        1. Summary
        2. Importance Score (1-10)
        3. Business Impact
        """

        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        analysis_text = response.content

        analyzed_results.append({

            "source": item.get("url", "Unknown"),

            "analysis": analysis_text
        })

    state["analyzed_data"] = analyzed_results

    add_log(
        state,
        f"Analyzed {len(analyzed_results)} sources."
    )

    return state


# ============================================================
# NODE 4 — QUALITY EVALUATION
# ============================================================

def quality_evaluation_node(state):

    add_log(state, "Evaluating research quality.")

    quality_prompt = f"""
    Evaluate the overall quality of this research.

    Research:
    {state["analyzed_data"]}

    Return ONLY a score between 1 and 10.
    """

    response = llm.invoke([
        HumanMessage(content=quality_prompt)
    ])

    try:
        score = int(response.content.strip())
    except:
        score = 5

    state["quality_score"] = score

    add_log(
        state,
        f"Research quality score = {score}"
    )

    return state


# ============================================================
# DYNAMIC ROUTING FUNCTION
# ============================================================

def dynamic_router(state):

    score = state["quality_score"]

    if score < 7:

        print("Quality insufficient -> recollecting data.")

        return "research_collection"

    else:

        print("Quality acceptable -> generating report.")

        return "report_generation"


# ============================================================
# NODE 5 — REPORT GENERATION
# ============================================================

def report_generation_node(state):

    add_log(state, "Generating enterprise research report.")

    report_prompt = f"""
    Generate a professional enterprise research report.

    Topic:
    {state["topic"]}

    Research Analysis:
    {state["analyzed_data"]}

    The report must include:
    - Executive Summary
    - Key Findings
    - Risks
    - Opportunities
    - Strategic Recommendations
    """

    response = llm.invoke([
        HumanMessage(content=report_prompt)
    ])

    state["final_report"] = response.content

    add_log(state, "Final report generated.")

    return state


# ============================================================
# NODE 6 — AUDIT LOGGING
# ============================================================

def audit_node(state):

    add_log(state, "Enterprise audit completed.")

    add_log(state, "Workflow execution finished.")

    return state


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node(
    "research_collection",
    research_collection_node
)

workflow.add_node(
    "memory_storage",
    memory_storage_node
)

workflow.add_node(
    "analysis",
    analysis_node
)

workflow.add_node(
    "quality_evaluation",
    quality_evaluation_node
)

workflow.add_node(
    "report_generation",
    report_generation_node
)

workflow.add_node(
    "audit",
    audit_node
)


# ============================================================
# GRAPH EDGES
# ============================================================

workflow.set_entry_point("research_collection")

workflow.add_edge(
    "research_collection",
    "memory_storage"
)

workflow.add_edge(
    "memory_storage",
    "analysis"
)

workflow.add_edge(
    "analysis",
    "quality_evaluation"
)

workflow.add_conditional_edges(
    "quality_evaluation",
    dynamic_router,
    {
        "research_collection": "research_collection",
        "report_generation": "report_generation"
    }
)

workflow.add_edge(
    "report_generation",
    "audit"
)

workflow.add_edge(
    "audit",
    END
)


# ============================================================
# COMPILE GRAPH
# ============================================================

app = workflow.compile()


# ============================================================
# INITIAL STATE
# ============================================================

initial_state = {

    "topic": "Enterprise Agentic AI Systems",

    "collected_data": [],

    "analyzed_data": [],

    "final_report": "",

    "execution_logs": [],

    "quality_score": 0
}


# ============================================================
# RUN ENTERPRISE AI AGENT
# ============================================================

final_state = app.invoke(initial_state)


# ============================================================
# DISPLAY FINAL REPORT
# ============================================================

print("\n")
print("================================================")
print("FINAL ENTERPRISE RESEARCH REPORT")
print("================================================")

print(final_state["final_report"])


# ============================================================
# DISPLAY EXECUTION LOGS
# ============================================================

print("\n")
print("================================================")
print("EXECUTION LOGS")
print("================================================")

for log in final_state["execution_logs"]:

    print(log)