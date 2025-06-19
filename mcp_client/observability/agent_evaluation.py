import importlib
import pandas as pd
from typing import Dict, Any
import re
import numpy as np
import cohere
import os

# === COHERE SETUP ===
api_key_prod = "uUlulV3HkN4ti01lrNIS6rwYgHoPkKInUoWVLBjr"
co = cohere.Client(api_key_prod)


# === EVALUATION FUNCTIONS ===

def exact_match(output: str, reference: str) -> bool:
    return output.strip() == reference.strip()


def custom_keyword_check(output: str, keyword_patterns: list) -> bool:
    return all(re.search(rf"\b({pattern})\b", output, re.IGNORECASE) for pattern in keyword_patterns)


def semantic_similarity_cohere(text1: str, text2: str, model="embed-english-v3.0") -> float:
    try:
        response = co.embed(
            texts=[text1, text2],
            model=model,
            input_type="search_document"  # REQUIRED for v3
        )
        embeddings = response.embeddings
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return round(float(cosine_sim), 4)
    except Exception as e:
        print(f"⚠️ Error computing semantic similarity: {e}")
        return 0.0


# === MAIN EVALUATION RUNNER ===

def evaluate_agent(agent_module_name: str, input_state: Dict[str, Any],
                   expected_output: str, output_field: str, keywords: list):
    result_row = {"Agent": agent_module_name}

    try:
        module = importlib.import_module(agent_module_name)
        agent = module.agent
    except Exception as e:
        result_row.update({
            "Semantic Similarity": "ImportError",
            "Exact Match": "ImportError",
            "Keyword Match": "ImportError",
            "Error": str(e)
        })
        return result_row

    try:
        result = agent.invoke(input_state)
        output = result.get(output_field, "")
        result_row["Semantic Similarity"] = semantic_similarity_cohere(output, expected_output)
        result_row["Exact Match"] = exact_match(output, expected_output)
        result_row["Keyword Match"] = custom_keyword_check(output, keywords)
        result_row["Generated Output"] = output
    except Exception as e:
        result_row.update({
            "Semantic Similarity": "RuntimeError",
            "Exact Match": "RuntimeError",
            "Keyword Match": "RuntimeError",
            "Error": str(e)
        })

    return result_row


# === TEST CASES ===

test_cases = [
    {
        "agent_module_name": "observability.langgraph_customer_support_agent",
        "input_state": {
            "issue": "battery not charging",
            "product": "wireless headphones",
            "step": "resolve"
        },
        "expected_output": "Please try a different charging cable or port. If the issue persists, your wireless headphones may need a battery replacement.",
        "output_field": "resolution",

        "keywords": ["battery|power", "charging|plug|usb", "headphones|earbuds|device"]

    },
    {
        "agent_module_name": "observability.langgraph_finance_advisor_agent",
        "input_state": {
            "goal": "save for child education",
            "risk_profile": "low",
            "step": "suggest_plan"
        },
        "expected_output": "For a low-risk profile aiming to save for child education, consider recurring deposits and low-risk mutual funds.",
        "output_field": "suggestion",

        "keywords": [
            "education|school|college",
            "low[- ]?risk|conservative|secure",
            "mutual funds|index funds|investment fund"
        ]
    }
]

# === EXECUTION ===

results = [evaluate_agent(**case) for case in test_cases]

# === SAVING REPORT AS CSV FILE ===

df = pd.DataFrame(results)
print("\n📊 LangGraph Agent Accuracy Evaluation Report:\n")
print(df.to_csv("LGEvaluationReport.csv", index=False))

# === PLAIN TEXT OUTPUT ===

print("\n📋 LangGraph Agent Evaluation Summary\n" + "-" * 60)
for result in results:
    print(f"\n🔹 Agent: {result.get('Agent', 'Unknown')}")
    if "Error" in result:
        print(f"❌ Error: {result['Error']}")
        continue
    print(f"🔁 Semantic Similarity : {result['Semantic Similarity']}")
    print(f"✅ Exact Match         : {result['Exact Match']}")
    print(f"🔍 Keyword Match       : {result['Keyword Match']}")
    print(f"📨 Generated Output:\n{result['Generated Output']}")
    print("-" * 60)