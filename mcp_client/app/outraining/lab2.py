from common.prompts import *
from debugpy.launcher.debuggee import describe
from llm.oci_genai import initialize_llm
from langchain.prompts import PromptTemplate
import os
from pathlib import Path

# python3.13 -m pip install openpyxl


llm = initialize_llm()

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent

########## Task 1
inputToPrompt1= "Explain the key features and benefits of the financial product: {card_name}."

describe_card_prompt = PromptTemplate.from_template(inputToPrompt1)
step1 = {'card_name': lambda line:line['inputFromUser']} | describe_card_prompt | llm


inputToPrompt2 = "Summarize this financial product in one compelling business-friendly line:\n\n{text}"
summary_tagline_prompt = PromptTemplate.from_template(inputToPrompt2)

step2 = {'text':step1} | summary_tagline_prompt | llm

response = step2.invoke({'inputFromUser':'Premium Gold Credit Card'})

#print(response.content)

###################### Task 2

# == Step 1:Capture Customer Need
intake_prompt = PromptTemplate.from_template(
    "You’re a retail-banking assistant. A customer says:\n\n“{customer_query}”\n\nSummarize their primary financial need in one sentence."
)
intake_chain = {"customer_query": lambda x: x["customer_query"]} | intake_prompt | llm

# == Step 2: Recommend Products
recommend_prompt = PromptTemplate.from_template(
    "Based on the need “{need_summary}”, recommend up to three retail-banking products (e.g., savings account, personal loan, credit card). For each, give a one-line rationale."
)
recommend_chain = {"need_summary": intake_chain} | recommend_prompt | llm

# == Step 3: Generate Customer-Facing Message
message_prompt = PromptTemplate.from_template(
    "Craft a concise, benefit-focused message to the customer, weaving in the recommended products:\n\nRecommendations:\n{product_recs}"
)
message_chain = {"product_recs": recommend_chain} | message_prompt | llm

# == Step 4: Draft Executive Summary
exec_prompt = PromptTemplate.from_template(
    """
Prepare a brief executive summary for senior leadership:
• Customer Need: {need_summary}
• Products Recommended: {product_recs}
• Customer Message Preview: {customer_message}
• Next Steps and KPIs to monitor
"""
)
full_chain = {
    "customer_query": lambda x: x["customer_query"],
    "need_summary": intake_chain,
    "product_recs": recommend_chain,
    "customer_message": message_chain
} | exec_prompt | llm

# == Execute the 4-step chain

output = full_chain.invoke({
    "customer_query": "I want to consolidate my credit-card debt and start earning travel rewards."
})

# == Display the generated output

print("Final Output:\n", output.content)

#################### Task 3




