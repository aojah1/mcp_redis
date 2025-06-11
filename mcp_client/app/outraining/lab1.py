from common.prompts import *
from debugpy.launcher.debuggee import describe
from llm.oci_genai import initialize_llm
from langchain.prompts import PromptTemplate
import os
from pathlib import Path
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

llm = initialize_llm()

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent


############# Task 1
# Correct f-string for file path
file_path = f"{PROJECT_ROOT}/Project_Y_Meeting_Transcript_June_3_2025.txt"


# Read the file content
with open(file_path, "r", encoding="utf-8") as file:
    meeting_transcript = file.read()

# variable `meeting_transcript` holds the entire transcript text
# You can use this variable as part of a prompt

#prompt = f"Based on the following meeting transcript, summarize the main decisions made:\n\n{meeting_transcript}"
prompt = PromptTemplate.from_template(prompt_example)
chain = prompt | llm

response = chain.invoke({'meeting_transcript',meeting_transcript})
#response.pretty_print()

############# Task 2

sys_prompt_ = "As a branding expert create a compelling story for product line {product_line}"
prompt = PromptTemplate.from_template(sys_prompt_)
user_prompt = "Oracle 23.ai"

chain = prompt | llm
response = chain.invoke({'product_line', user_prompt})
#response.pretty_print()

########### Task 3

sys_prompt_ = """
You are a customer service representative at a bank.
Write a {tone}  follow-up email with the subject: {subject}.
use the following customer interaction summary to generate the message:\n\n{context}\n\nEmail:"""

prompt = PromptTemplate.from_template(sys_prompt_)
chain = prompt | llm

input_data = {
    'tone': 'empethatic and professional',
    'subject': 'Update on your Recent Transaction Dispute',
    'context': """The customer reported an unauthorized debit of ₹5,000 on June 8 from their savings account. "
        "They confirmed they did not initiate the transaction. We informed them that an internal investigation has been initiated, "
        "and the resolution timeline is 3–5 business days as per bank policy."""
}

response = chain.invoke(input_data)
#response.pretty_print()

############## Task 4

# Load multiple inputs from Excel
df = pd.read_excel(f"{PROJECT_ROOT}/banking_email_input_multiple.xlsx")
print(df.head(10))
# Define prompt template

prompt = PromptTemplate.from_template(
    """You are a customer service representative at a bank.
Write a {tone} follow-up email with the subject: "{subject}".
Use the following customer interaction summary to generate the message:\n\n{context}\n\nEmail:"""
)

# Create LangChain pipeline
chain = prompt | llm

# Process each row and store responses
email_outputs = []
for _, row in df.iterrows():
    input_data = row.to_dict()
    email = chain.invoke(input_data)
    email_outputs.append(email)

# Add results to the dataframe and export
df["Generated_Email"] = email_outputs
output_path = f"{PROJECT_ROOT}/banking_email_output.xlsx"
df.to_excel(output_path, index=False)

print(f"Batch email generation completed. Output saved to: {output_path}")

############# Task 5


# SMTP email configuration (Gmail example)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "meghatse@gmail.com"        #"your_email@gmail.com"
SENDER_PASSWORD = "xjbplfyhwtbyznps"         # "your_app_password"  # Use an app-specific password
RECIPIENT_EMAIL = "anup.ojah@oracle.com"      # "recipient@example.com"  # Can be dynamic per row if needed

# === Load Excel Input ===
df = pd.read_excel(f"{PROJECT_ROOT}/banking_email_input_multiple.xlsx")

# === Define Prompt ===

prompt = PromptTemplate.from_template(
    """You are a customer service representative at a bank.
Write a {tone} follow-up email with the subject: "{subject}".
Use the following customer interaction summary to generate the message:\n\n{context}\n\nEmail:"""
)

chain = prompt | llm

# === Generate and Send Emails ===
generated_emails = []
html_emails = []
smtp_statuses = []

for _, row in df.iterrows():
    input_data = row.to_dict()

    # Generate plain email
    # plain_email = chain.invoke(input_data)

# Invoke the language model with the filled prompt
    response = chain.invoke(input_data)
    plain_email = response.content
    generated_emails.append(plain_email)

    # Convert to HTML
    body_html = plain_email.replace("\n", "<br>")
    context_html = row["context"].replace("\n", "<br>")
    html_email = (
        f"<html><body>"
        f"<p><strong>Subject:</strong> {row['subject']}</p>"
        f"<p><strong>Tone:</strong> {row['tone']}</p>"
        f"<p><strong>Customer Context:</strong><br>{context_html}</p><hr>"
        f"<p>{body_html}</p></body></html>"
    )
    html_emails.append(html_email)

    # Prepare MIME email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = row["subject"]
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg.attach(MIMEText(html_email, "html"))

    # Send email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            #server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        smtp_statuses.append("Sent successfully")
    except Exception as e:
        smtp_statuses.append(f"Failed to send: {str(e)}")

# === Save Output ===
df["Generated_Email"] = generated_emails
df["HTML_Email"] = html_emails
df["SMTP_Status"] = smtp_statuses
df.to_excel(f"{PROJECT_ROOT}/banking_email_output_with_html_smtp_sent.xlsx", index=False)

print("All emails processed. Results saved to 'banking_email_output_with_html_smtp_sent.xlsx'")

############# Task 6

# Define a dynamic prompt to personalize claim explanations

# Jinja2-based dynamic prompt template

template_str = """
You are an AI assistant at a financial services company helping customers understand their insurance claims.

{% if customer_type == "premium" %}
Respond in a professional, detailed tone with appreciation for customer loyalty.
{% else %}
Respond in a clear, supportive tone with helpful instructions.
{% endif %}

Claim Type: {{ claim_type }}

{% if claim_type == "health" %}
Explain the health insurance claim process, required documents, and expected timeline.
{% elif claim_type == "vehicle" %}
Describe how to file a vehicle insurance claim, inspection steps, and settlement terms.
{% elif claim_type == "life" %}
Outline the steps for a life insurance claim including nominee verification and required forms.
{% else %}
Advise the customer to contact support for claim type-specific instructions.
{% endif %}
"""


# Create a PromptTemplate with Jinja2 formatting

prompt = PromptTemplate.from_template(template_str, template_format="jinja2")


# Define user-specific input for the claim type and customer type
input_variables = {
    "claim_type": "vehicle",        # Options: "health", "vehicle", "life", or others
    "customer_type": "regular"      # Options: "premium" or "regular"
}

#  Format the prompt and call the LLM

filled_prompt = prompt.format(**input_variables)
print("Filled Prompt:\n", filled_prompt)
print("=======================================================================\n")

# Invoke the language model with the filled prompt

response = llm.invoke(filled_prompt)

response.pretty_print()