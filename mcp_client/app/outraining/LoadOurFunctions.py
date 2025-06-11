def get_CohereAIResponse(prompt):
    import oci
    from LoadProperties import LoadProperties
    properties=LoadProperties()
    
    chat_request = oci.generative_ai_inference.models.CohereChatRequest()
    chat_request.max_tokens=600
    chat_request.temperature=0.01
    chat_request.message=prompt
    
    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_detail.serving_mode= oci.generative_ai_inference.models.OnDemandServingMode(
        model_id=properties.getModelName())  # cohere.command-r-08-2024
    chat_detail.compartment_id=properties.getCompartment()
    chat_detail.chat_request=chat_request

    signer=oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    # Service endpoint
    generative_ai_inference_client=oci.generative_ai_inference.GenerativeAiInferenceClient(
    config={},  signer=signer,  
    service_endpoint=properties.getEndpoint(),
    retry_stratergy=oci.retry.NoneRetryStrategy(),
    timeout=(10,240))
    
    chat_response = generative_ai_inference_client.chat(chat_detail)

    # print("***************Response From AI***********")
    # print()
    # print(chat_response.data.chat_response.text)
    output = chat_response.data.chat_response.text
    return output
    



def get_LlamaAIResponse(prompt):
    import oci
    from LoadProperties import LoadProperties
    properties=LoadProperties()

    
    content = oci.generative_ai_inference.models.TextContent()
    content.text = "{0}".format(prompt)
    
    message = oci.generative_ai_inference.models.Message()
    message.role = "USER"
    message.content = [content]
    
    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    
    chat_request.messages = [message]
    chat_request.max_tokens=600
    chat_request.temperature=0.01
    
    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        model_id='meta.llama-3.3-70b-instruct')
    
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = properties.getCompartment()
    
    
    signer=oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    # Service endpoint
    generative_ai_inference_client=oci.generative_ai_inference.GenerativeAiInferenceClient(
        config={},  signer=signer,  
        service_endpoint=properties.getEndpoint(),
        retry_stratergy=oci.retry.NoneRetryStrategy(),
        timeout=(10,240)
    )
    
    chat_response = generative_ai_inference_client.chat(chat_detail)
    
    # Print result
    # print("**************************Chat Result**************************")
    # print(vars(chat_response))
    # print(vars(vars(vars((vars(chat_response.data.chat_response)['_choices'][0]))['_message'])['_content'][0])['_text'])

    output = (vars(vars(vars((vars(chat_response.data.chat_response)['_choices'][0]))['_message'])['_content'][0])['_text'])

    return output

    
    