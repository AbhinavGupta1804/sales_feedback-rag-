import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def initialize_openai():
    """
    Initializes and returns the LangChain OpenAI LLM
    using environment variables.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4.1-mini"

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.3,
        api_key=api_key
    )

    return llm


if __name__ == "__main__":
    llm = initialize_openai()

    response = llm.invoke("Summarize a sales call in one sentence.")
    
    # LangChain returns an AIMessage object
    finalans = response.content.strip()
    print(finalans)


# import os
# import json
# import time
# import boto3
# from dotenv import load_dotenv
# from botocore.exceptions import ClientError


# # -----------------------------
# # CONFIGmodel_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
# # -----------------------------
# DEFAULT_REGION = "us-east-1"
# DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
# MAX_RETRIES = 1
# SAFE_TOTAL_TOKENS = 8000   # Hard safety guard per request


# # -----------------------------
# # UTILS
# # -----------------------------
# def estimate_tokens(text: str) -> int:
#     """
#     Rough token estimator:
#     1 token ≈ 0.75 words
#     """
#     return int(len(text.split()) / 0.75)


# # -----------------------------
# # INIT
# # -----------------------------
# def initialize_bedrock():
#     """
#     Initializes AWS Bedrock runtime client
#     """
#     load_dotenv()

#     region = os.getenv("AWS_REGION", DEFAULT_REGION)

#     return boto3.client(
#         service_name="bedrock-runtime",
#         region_name=region,
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     )


# # -----------------------------
# # MAIN CALL
# # -----------------------------
# def generate_llm_response(prompt, max_tokens=500, temperature=0.3):
#     """
#     Safe, production-style Claude 4.5 Haiku call
#     """

#     # --- TOKEN SAFETY CHECK ---
#     estimated = estimate_tokens(prompt) + max_tokens
#     if estimated > SAFE_TOTAL_TOKENS:
#         raise ValueError(
#             f"Token safety triggered: {estimated} > {SAFE_TOTAL_TOKENS}. "
#             "Trim RAG context or lower max_tokens."
#         )

#     client = initialize_bedrock()
#     model_id = os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL)

#     body = {
#         "anthropic_version": "bedrock-2023-05-31",
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     }

#     attempt = 0
#     while attempt <= MAX_RETRIES:
#         try:
#             response = client.invoke_model(
#                 modelId=model_id,
#                 contentType="application/json",
#                 accept="application/json",
#                 body=json.dumps(body)
#             )

#             result = json.loads(response["body"].read())
#             return result["content"][0]["text"]

#         except ClientError as e:
#             error_code = e.response["Error"]["Code"]

#             # Rate limit or quota issues
#             if error_code in ["ThrottlingException", "LimitExceededException"]:
#                 if attempt >= MAX_RETRIES:
#                     raise RuntimeError(
#                         "Bedrock rate limit hit. "
#                         "Try switching region or wait for quota reset."
#                     )

#                 wait_time = 2 ** attempt
#                 time.sleep(wait_time)
#                 attempt += 1
#                 continue

#             # Other AWS errors
#             raise RuntimeError(f"Bedrock error: {str(e)}")


# # -----------------------------
# # QUICK TEST
# # -----------------------------
# if __name__ == "__main__":
#     test_prompt = "Summarize a sales call in one sentence."
#     print(generate_llm_response(test_prompt))

# import os
# import google.generativeai as genai
# from dotenv import load_dotenv


# def initialize_gemini():
#     """
#     Initializes and returns the Gemini LLM model
#     using environment variables.
#     """

#     load_dotenv()

#     api_key = os.getenv("GEMINI_API_KEY")
#     model_name = "gemini-2.5-flash"

#     if not api_key:
#         raise ValueError("GEMINI_API_KEY not found in .env")

#     genai.configure(api_key=api_key)

#     llm = genai.GenerativeModel(model_name)

#     return llm

# # import boto3
# # from langchain_aws import BedrockLLM


# # def initialize_bedrock_llm():
# #     AWS_REGION = "us-east-1"
# #     BEDROCK_MODEL_ID = "google.gemma-3-12b-it"

# #     bedrock_client = boto3.client(
# #         service_name="bedrock-runtime",
# #         region_name=AWS_REGION
# #     )

# #     llm = BedrockLLM(
# #         client=bedrock_client,
# #         model_id=BEDROCK_MODEL_ID,
# #         model_kwargs={
# #             "temperature": 0.3,
# #             "max_tokens": 300
# #         }
# #     )

# #     return llm




