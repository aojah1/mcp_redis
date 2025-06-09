from langchain_community.chat_models import ChatOCIModelDeployment
import oci
import ads
AUTH_TYPE      = "api_key"
CONFIG_PROFILE = "DEFAULT"

def initialize_llm():
    ads.set_auth(AUTH_TYPE)
    llm_md = ChatOCIModelDeployment(
        model = "odsc-llm",
        endpoint = "https://modeldeployment.ap-osaka-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.ap-osaka-1.amaaaaaawe6j4fqaeclsuxaga4bslar4cpqyjomjckvykgk7ecpxb7moheia/predict",
        max_tokens = 1024,
        streaming = True,
        auth_profile=CONFIG_PROFILE,
        enable_auto_tool_choice=True,
        tool_call_parser = "llama3_json",
        tool_choice="auto"
    )

    # Invocation
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    response = llm_md.invoke(messages)
    print(response.content)

    return llm_md

# import os
# from oci.signer import Signer
# from langchain_community.chat_models import ChatOCIModelDeployment
#
# #────────────────────────────────────────────────────────
# # OCI GenAI configuration
# # ────────────────────────────────────────────────────────
#
# AUTH_TYPE      = "API_KEY"
# CONFIG_PROFILE = "DEFAULT"
#
#
# def initialize_llm() -> ChatOCIModelDeployment:
#     # 1) Load your OCI API-key credentials from env (or ~/.oci/config)
#     model_id   = os.getenv("OCI_MODEL_ID_1")
#
#     # 1) Load your OCI config (defaults to ~/.oci/config [DEFAULT])
#     config = oci.config.from_file(
#         file_location=os.path.expanduser("~/.oci/config"),
#         profile_name=os.getenv("OCI_PROFILE", "DEFAULT")
#     )
#
#     # 2) Construct the APIKeySigner
#     signer = Signer(
#         tenancy=config["tenancy"],
#         user=config["user"],
#             fingerprint=config["fingerprint"],
#         private_key_file_location=config["key_file"],
#         pass_phrase=config.get("pass_phrase", None),
#         #region=config["region"]
#     )
#
#     # 3) Your GenAI endpoint (replace or set via env)
#     endpoint   = os.getenv("OCI_LLM_ENDPOINT")
#
#     # 4) Instantiate the LLM, supplying only the signer
#     return ChatOCIModelDeployment(
#         model=model_id,
#         endpoint=endpoint,
#         auth={"signer": signer},
#         default_headers={"route": "v1/chat/completions"},
#     )
if __name__ == "__main__":
    initialize_llm()

