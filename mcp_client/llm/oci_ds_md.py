import ads
from langchain_community.chat_models import ChatOCIModelDeployment

llm_md = ChatOCIModelDeployment(
    model = "odsc-llm",
    endpoint = "https://modeldeployment.ap-osaka-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.ap-osaka-1.amaaaaaawe6j4fqamxfjgxik5utnjp6v3nhbwgqroakj5nr73zeu6rqikkqa/predict",
    max_tokens = 1024,
    streaming = True,
    enable_auto_tool_choice=True,
    tool_call_parser=True
)