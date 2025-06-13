#  Enhanced prompt , parameters to improve score
import time
import requests
#import cohere
from llm.oci_genai import initialize_llm
import re
import pandas as pd
from functools import lru_cache
from rouge_score import rouge_scorer

# === CONFIGURATION ===
cohere_client = initialize_llm()


## === TOOL 1: Wikipedia Search Tool ===
class WikipediaSearchTool:
    @lru_cache(maxsize=64)
    def search(self, query):
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                return res.json().get("extract", "No summary available.")
            return f"⚠️ Wikipedia returned status {res.status_code}"
        except Exception as e:
            return f"❌ Wikipedia error: {e}"

# === TOOL 2: OpenLibrary Search Tool ===
class OpenLibrarySearchTool:
    @lru_cache(maxsize=64)
    def search(self, query):
        url = f"http://openlibrary.org/search.json?q={query}"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                docs = res.json().get("docs", [])
                if docs:
                    titles = [book.get("title", "Unknown Title") for book in docs[:3]]
                    return "Top books:\n- " + "\n- ".join(titles)
                return "No books found."
            return f"⚠️ OpenLibrary returned status {res.status_code}"
        except Exception as e:
            return f"❌ OpenLibrary error: {e}"

# === AGENT ===
class MultiToolAgent:
    def __init__(self):
        self.wikipedia = WikipediaSearchTool()
        self.openlibrary = OpenLibrarySearchTool()

    def select_tool(self, query):
        q = query.lower()
        if any(x in q for x in ["book", "read", "novel"]):
            return self.openlibrary.search, "openlibrary"
        if re.match(r"(?i)^(tell me about|what is|who is|define|explain)", q.strip()):
            return self.wikipedia.search, "wikipedia"
        return None, None

    def prompt_engineering(self, context, question):
        return f"""You are a knowledgeable assistant. Provide a factual, well-structured answer based only on the context below.


{context}


Question: {question}

Answer (brief and informative):"""

    def generate_answer(self, context, question):
        prompt = self.prompt_engineering(context, question)
        try:
            response = cohere_client.generate(
                model="command-r-plus",
                prompt=prompt,
                max_tokens=150,
                temperature=0.3,
                k=0,
                stop_sequences=["\n\n"]
            )
            return response.generations[0].text.strip()
        except Exception as e:
            return f"❌ Cohere error: {e}"

    def answer(self, query):
        tool_func, tool_name = self.select_tool(query)
        if not tool_func:
            return "🤷 Sorry, no matching tool available."
        cleaned_query = re.sub(r"(?i)^tell me about|^what is|^who is|^define|^explain|^recommend( some)? books( to read)?( on| about)?|^suggest( some)? books( on| about)?", "", query).strip()
        context = tool_func(cleaned_query)
        if "⚠️" in context or "❌" in context:
            return f"⚠️ Failed to retrieve data using {tool_name}."
        return self.generate_answer(context, query)

# === NORMALIZATION + TESTING ===
def normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def test_agent_performance(agent, queries, expected_keywords):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = []
    rouge_scores = []

    for i, (query, expected) in enumerate(zip(queries, expected_keywords), 1):
        start = time.time()
        answer = agent.answer(query)
        latency = round(time.time() - start, 2)

        ref = normalize(" ".join(expected))
        pred = normalize(answer)
        rouge = scorer.score(ref, pred)['rougeL'].fmeasure if ref else 0.0
        rouge_scores.append(rouge)

        keyword_match = 1 if any(k.lower() in pred for k in expected) else 0

        results.append({
            "Query #": i,
            "Query": query,
            "Answer": answer,
            "Latency (s)": latency,
            "Accurate?": "✅" if keyword_match else "❌",
            "ROUGE-L": round(rouge, 4)
        })

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    return results, avg_rouge

# === SAMPLE QUERIES ===
# ==== Call the Agent ===
agent = MultiToolAgent()

sample_queries = [
    "Tell me about Alan Turing",
    "What is quantum computing?",
    "Recommend some books on artificial intelligence",
    "Suggest books to read on mental health",
    "Who is Ada Lovelace?",
    "Explain blockchain technology",
    "How do I install TensorFlow?"
]
expected_keywords = [
    ["turing", "enigma", "computing"],
    ["quantum", "entanglement", "qubits"],
    ["artificial", "intelligence", "learning"],
    ["mental", "health", "psychology"],
    ["lovelace", "programmer", "analytical"],
    ["blockchain", "ledger", "distributed"],
    []
]

# === RUN ===
results, avg_rouge_score = test_agent_performance(agent, sample_queries, expected_keywords)

print("\n=== Agent Query Results ===")
for row in results:
    print(f"\n🔹 Query #{row['Query #']}: {row['Query']}")
    print(f"🕒 Latency: {row['Latency (s)']} seconds")
    print(f"🎯 Accurate?: {row['Accurate?']}")
    print(f"📏 ROUGE-L Score: {row['ROUGE-L']}")
    print(f"🧠 Answer:\n{row['Answer']}")

prev_run_score = avg_rouge_score
print(f"\n📊 Final ROUGE-L Accuracy Score: {avg_rouge_score:.4f}")


# === TOOL 1: Wikipedia Search Tool ===
class WikipediaSearchTool:
    @lru_cache(maxsize=64)
    def search(self, query):
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                return res.json().get("extract", "No summary available.")
            return f"⚠️ Wikipedia returned status {res.status_code}"
        except Exception as e:
            return f"❌ Wikipedia error: {e}"


# === TOOL 2: OpenLibrary Search Tool ===
class OpenLibrarySearchTool:
    @lru_cache(maxsize=64)
    def search(self, query):
        url = f"http://openlibrary.org/search.json?q={query}"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                docs = res.json().get("docs", [])
                if docs:
                    titles = [book.get("title", "Unknown Title") for book in docs[:3]]
                    return "Top books:\n- " + "\n- ".join(titles)
                return "No books found."
            return f"⚠️ OpenLibrary returned status {res.status_code}"
        except Exception as e:
            return f"❌ OpenLibrary error: {e}"


# === AGENT ===
class MultiToolAgent:
    def __init__(self):
        self.wikipedia = WikipediaSearchTool()
        self.openlibrary = OpenLibrarySearchTool()

    def select_tool(self, query):
        q = query.lower()
        if any(x in q for x in ["book", "read", "novel"]):
            return self.openlibrary.search, "openlibrary"
        if re.match(r"(?i)^(tell me about|what is|who is|define|explain)", q.strip()):
            return self.wikipedia.search, "wikipedia"
        return None, None

    def prompt_engineering(self, context, question):
        # Optimized prompt to improve factual alignment and phrasing for ROUGE-L
        return f"""You are a highly factual and precise assistant. Use only the information provided below to answer the question. Do not add assumptions.

    Context:
    \"\"\"
    {context}
    \"\"\"

    Question: {question}

    Answer:
    """

    def generate_answer(self, context, question):
        prompt = self.prompt_engineering(context, question)
        try:

            # LLM configuration for deterministic, accurate output
            response = cohere_client.generate(
                model="command-r-plus",
                prompt=prompt,
                max_tokens=200,  # 🟢 Increase to allow full factual phrasing
                temperature=0.2,  # 🟢 Lower = more deterministic = higher ROUGE
                p=0.8,  # 🆕 Limit top-p nucleus sampling for safe diversity
                k=0,  # Deterministic mode
                stop_sequences=["\n\n", "\nQuestion:"]
            )

            return response.generations[0].text.strip()
        except Exception as e:
            return f"❌ Cohere error: {e}"

    def answer(self, query):
        tool_func, tool_name = self.select_tool(query)
        if not tool_func:
            return "🤷 Sorry, no matching tool available."
        # Standardized query cleaning
        cleaned_query = re.sub(
            r"(?i)^tell me about|^what is|^who is|^define|^explain|^recommend( some)? books( to read)?( on| about)?|^suggest( some)? books( on| about)?",
            "", query).strip()
        context = tool_func(cleaned_query)
        if "⚠️" in context or "❌" in context:
            return f"⚠️ Failed to retrieve data using {tool_name}."
        return self.generate_answer(context, query)


# === SUPPORT FUNCTIONS ===
def normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def test_agent_performance(agent, queries, expected_keywords):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = []
    rouge_scores = []

    for i, (query, expected) in enumerate(zip(queries, expected_keywords), 1):
        start = time.time()
        answer = agent.answer(query)
        latency = round(time.time() - start, 2)

        ref = normalize(" ".join(expected))
        pred = normalize(answer)
        rouge = scorer.score(ref, pred)['rougeL'].fmeasure if ref else 0.0
        rouge_scores.append(rouge)

        keyword_match = 1 if any(k.lower() in pred for k in expected) else 0

        results.append({
            "Query #": i,
            "Query": query,
            "Answer": answer,
            "Latency (s)": latency,
            "Accurate?": "✅" if keyword_match else "❌",
            "ROUGE-L": round(rouge, 4)
        })

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    return results, avg_rouge


# === INPUTS ===
agent = MultiToolAgent()
sample_queries = [
    "Tell me about Alan Turing",
    "What is quantum computing?",
    "Recommend some books on artificial intelligence",
    "Suggest books to read on mental health",
    "Who is Ada Lovelace?",
    "Explain blockchain technology",
    "How do I install TensorFlow?"
]
expected_keywords = [
    ["turing", "enigma", "computing"],
    ["quantum", "entanglement", "qubits"],
    ["artificial", "intelligence", "learning"],
    ["mental", "health", "psychology"],
    ["lovelace", "programmer", "analytical"],
    ["blockchain", "ledger", "distributed"],
    []
]

# === EXECUTION ===
results, avg_rouge_score = test_agent_performance(agent, sample_queries, expected_keywords)

# === OUTPUT ===
print("\n=== Agent Query Results (Enhanced Prompt + LLM Settings) ===")
for row in results:
    print(f"\n🔹 Query #{row['Query #']}: {row['Query']}")
    print(f"🕒 Latency: {row['Latency (s)']} seconds")
    print(f"🎯 Accurate?: {row['Accurate?']}")
    print(f"📏 ROUGE-L Score: {row['ROUGE-L']}")
    print(f"🧠 Answer:\n{row['Answer']}")

print(f"\n🎯 Final ROUGE-L Accuracy Score: {avg_rouge_score:.4f}")
print(f"\n📊 Previous ROUGE-L Accuracy Score: {prev_run_score:.4f}")

