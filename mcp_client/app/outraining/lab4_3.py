from common.prompts import *
from debugpy.launcher.debuggee import describe
from llm.oci_genai import initialize_llm
from llm.oci_embedding_model import initialize_embedding_model
from langchain.prompts import PromptTemplate
import os
from pathlib import Path
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.utilities import WikipediaAPIWrapper
# python3.13 -m pip install openpyxl


llm = initialize_llm()
embeddding_model = initialize_embedding_model()

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent


# Tools
calculator = lambda x: str(eval(x)) if x.strip() else "Invalid"
get_founder = lambda x: {"Apple": "Steve Jobs", "Microsoft": "Bill Gates", "Amazon": "Jeff Bezos", "Google": "Larry Page and Sergey Brin"}.get(x.strip(), "Founder not found.")
get_city_info = lambda x: {"Paris": "Paris is the capital of France.", "New York": "New York is known as the Big Apple.", "Tokyo": "Tokyo is the capital of Japan."}.get(x.strip(), "City info not found.")
tools = [
    Tool("Calculator", calculator, "Solve math like '45 * 3'."),
    Tool("CompanyFounderLookup", get_founder, "Find the founder of a company."),
    Tool("CityInformation", get_city_info, "Get facts about cities.")
]

# Output schema
parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="answer", description="Final answer."),
    ResponseSchema(name="source", description="Tool used.")
])
fmt = parser.get_format_instructions()


# Agent
system_prompt =f"""
You are a helpful agent that uses tools to answer user questions. Choose the correct tool to respond, and keep your answers concise and accurate.

You have access to:
- Calculator: Solve basic math like '45 * 3'
- CompanyFounderLookup: Find the founder of companies like Apple, Microsoft
- CityInformation: Get facts about cities like Paris, Tokyo

Here are some examples of how to answer:

Q: What is 45 * 3?
A: 135 (Calculator)

Q: Who founded Amazon?
A: Jeff Bezos (CompanyFounderLookup)

Q: Tell me something about Tokyo.
A: Tokyo is the capital of Japan. (CityInformation)

Always pick the most suitable tool and return only the final answer with the tool name in parentheses.
"""
agent = initialize_agent(
    tools=tools, llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, handle_parsing_errors=True,
    agent_kwargs={"system_message": system_prompt},
    max_iterations=4, early_stopping_method="generate"
)

# Run queries
queries = ["What is 45 * 3?", "Who founded Google?", "Tell me something about Paris."]
# Replace the loop with this:
for q in queries:
    print(f"\n {q}")
    res = agent.invoke({"input": q})
    answer = res.get("output", " No output")
    print(f"**Answer:** {answer}")


############# task 2

# --- Calculator Tool ---
def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error in calculation: {e}"

# --- Wikipedia Tool ---
wiki = WikipediaAPIWrapper()

# --- Tools ---
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for solving basic math expressions like '12*4' or 'sqrt(64)'."
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for answering factual questions using Wikipedia content."
    )
]

# --- Define structured JSON output schema ---
response_schemas = [
    ResponseSchema(name="answer", description="The final answer to the user's question"),
    ResponseSchema(name="source", description="The tool(s) or source(s) used for this answer")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# --- Custom system prompt to enforce JSON format ---
system_prompt = f"""
You are a helpful reasoning agent. Use tools like Calculator and Wikipedia to answer the question.

At the end, always respond ONLY in the following JSON format:

{format_instructions}

Do not include any explanation or formatting outside this JSON object.
"""


# --- Initialize Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs={"system_message": system_prompt}
)

# --- User Query ---
query = "What is 17 * 24 and who invented the telescope? Respond only with JSON keys 'answer' and 'source'."


# --- Run the agent ---
response = agent.invoke({"input": query})

parsed_output = output_parser.parse(response["output"])
print("\n✅ Final Parsed JSON Output:")
print(parsed_output)
print(f"**Answer:** {parsed_output['answer']}  \n**Source:** {parsed_output['source']}")


######## Task 3
# --- Define Tools ---
tools = [
    Tool("Calculator", lambda x: str(eval(x)), "Solve math like '17 * 24'."),
    Tool("Wikipedia", WikipediaAPIWrapper().run, "Answer factual questions from Wikipedia.")
]

# --- Output Schema + Parser ---
schemas = [
    ResponseSchema(name="answer", description="Final answer."),
    ResponseSchema(name="source", description="Tool(s) used.")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
fmt = parser.get_format_instructions()



# --- Use Reliable Agent Type for OCI ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs={"system_message": f"""
You are a helpful agent. Use Calculator or Wikipedia to answer questions.

Always include the tool(s) used in your final JSON response, under the 'source' key.

Return ONLY in this JSON format:
{fmt}

Examples:
{{"answer": "17 * 24 is 408", "source": "Calculator"}}
{{"answer": "Hans Lippershey invented the telescope", "source": "Wikipedia"}}
"""}
)

# --- Run Agent + Display as Markdown Table ---
res = agent.invoke({"input": "What is 17 * 24 and who invented the telescope?"})
try:
    out = parser.parse(res["output"])
except:
    # Minimal fallback: fix with the LLM if needed
    repaired = llm.invoke(f"""
Format this into JSON:
{fmt}

Output:
{res['output']}
""")
    out = parser.parse(repaired.content if hasattr(repaired, "content") else str(repaired))

md = f"""| Field   | Value |
|---------|-------|
| Answer  | {out['answer']} |
| Source  | {out['source']} |"""
print(md)




