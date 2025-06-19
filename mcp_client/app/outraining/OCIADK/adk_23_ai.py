from pypdf import PdfReader
import oracledb
import oci
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import BaseDocumentTransformer,Document
#from LoadProperties import LoadProperties
from llm.oci_embedding_model import initialize_embedding_model
from typing import Dict
from oci.addons.adk import Agent, AgentClient, tool
import oci,os
from dotenv import load_dotenv
from pathlib import Path

print("Successfully imported libraries and modules")

embed_model = initialize_embedding_model()

# sudo podman exec -i 23ai bash <<EOF
# sqlplus -S vector/vector@localhost:1521/freepdb1 <<EOSQL
# SELECT * FROM TAB;
# EXIT;
# EOSQL
# EOF

CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file(profile_name=CONFIG_PROFILE)  # Update this with your own profile name
sess_id = ""

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")
print(PROJECT_ROOT)
# Set up the OCI GenAI Agents endpoint configuration
AGENT_EP_ID = os.getenv("AGENT_EP_ID")
AGENT_SERVICE_EP = os.getenv("AGENT_SERVICE_EP")

un = "vector"
pw = "vector"
cs = "158.180.27.71:1521/FREEPDB1"


try:
    conn3c = oracledb.connect(user=un,password=pw,dsn=cs)
    print("Connection successful !!")
except Exception as e :
    print("Connection failed !!")

# RAG Step1 Load the PDF document and create pdf reader object

pdf = PdfReader(f'{PROJECT_ROOT}/oci-ai-foundations.pdf')

# RAG step2 Transform the document to text
text = ""

for page in pdf.pages:
    text += page.extract_text()

print("You have transformed the PDF document to text format")

# RAG step3 Chunk the text into smaller chunks

text_splitter = CharacterTextSplitter(separator=".", chunk_size=2000, chunk_overlap=100)

chunks = text_splitter.split_text(text)
print(len(chunks))

# Function to format and add metadata to Oracle 23ai Vector store

def chunks_to_docs_wrapper(row:dict) -> Document:
    metadata={'id':row['id'], 'link':row['link']}
    return Document(page_content=row['text'], metadata=metadata)

# RAG step4 create metadata wrapper to store additional information in vector store

docs = [chunks_to_docs_wrapper({'id':str(page_num),'link':
                               f'Page {page_num}', 'text':text}) for page_num, text in enumerate(chunks)]

# RAG step5 using an embedding model embed the chunks as vectors into oracle database 23ai

# RAG step6 configure the vector store with the model , table name and using indicated distance
# strategy for the similarity search and vectorize the chunks

knowledge_base = OracleVS.from_documents(docs, embed_model,client=conn3c,
                                        table_name="DEMO_TABLE_AO",
                                        distance_strategy=DistanceStrategy.DOT_PRODUCT)

print("Chunks are stored in the DEMO_TABLE_AO")

def run_queries(query:str):
    # Create a cursor from the connection
    cursor = conn3c.cursor()

    # Example: Run a SELECT query
    cursor.execute(query)

    # Fetch all results
    rows = cursor.fetchall()

    # Print the results
    for row in rows:
        print("sql response: ")
        print(row)

    # When done, close cursor and connection
    cursor.close()


run_queries("Select id,text,metadata from DEMO_TABLE  where id ='FF5A1AE012AFA5D4'")
# ==================Check table Existence=============
#
# sudo podman exec -i 23ai bash <<EOF
# sqlplus -S 158.180.27.71:1521/FREEPDB1 <<EOSQL
# SELECT * FROM DEMO_TABLE_AO;
# EXIT;
# EOSQL
# EOF
#
# ==================Describe table =============
#
# sudo podman exec -i 23ai bash <<EOF
# sqlplus -S vector/vector@localhost:1521/freepdb1 <<EOSQL
# DESCRIBE DEMO_TABLE;
# EXIT;
# EOSQL
# EOF
#
# ==================Count Rows in table =============
#
# sudo podman exec -i 23ai bash <<EOF
# sqlplus -S vector/vector@localhost:1521/freepdb1 <<EOSQL
# Select count(*) from DEMO_TABLE;
# EXIT;
# EOSQL
# EOF
#
# ==================Query the table =============
#
#
# sudo podman exec -i 23ai bash <<EOF
# sqlplus -S vector/vector@localhost:1521/freepdb1 <<EOSQL
# Select id,text,metadata from DEMO_TABLE  where id ='FF5A1AE012AFA5D4';
# EXIT;
# EOSQL
# EOF
#
# ==================Query the table and embedding column =============
#
# sudo podman exec -i 23ai bash <<EOF
# sqlplus -S vector/vector@localhost:1521/freepdb1 <<EOSQL
# Select id,text,metadata,EMBEDDING from DEMO_TABLE  where id ='FF5A1AE012AFA5D4';
# EXIT;
# EOSQL
# EOF

################# Task 2

#%% md
### RAG with `ADK Agent` + `Knowledge Base Oracle 23ai vector db`

#%% md
### Tool Creation to retrive docs from Oracle 23ai

@tool
def retrieve_documents(query: str) -> dict:
    """Retrieve course details from context"""

    vs = OracleVS(embedding_function=embed_model, client=conn3c,
                  table_name="DEMO_TABLE_AO",
                  distance_strategy=DistanceStrategy.DOT_PRODUCT)

    retv = vs.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    content_doc = retv.get_relevant_documents(query)
    raw_text = ''
    for i in range(len(content_doc)):
        raw_text += content_doc[i].page_content

    return {"content": raw_text}

#%% md
### Agent initalization and Tool registration with Agent
client = AgentClient(auth_type="api_key", profile=CONFIG_PROFILE, region="us-chicago-1")

agent = Agent(
    client=client,
        # Agent create on oci with name OCI-DEMO-AGENT-1
        agent_endpoint_id=AGENT_EP_ID,
    instructions="You are a smart assistant. Get information from context provided.",
    tools=[retrieve_documents]   # Tool Registration
)

agent.setup()

response = agent.run("what is oci ai foundation course") # Query1 To Ask

# response = agent.run("How many modules are in oci ai foundations course?")  # Query2 To Ask

print(response.data["message"]["content"]["text"])
