import os
from langchain_openai import ChatOpenAI

GPT_MODEL_NAME = os.getenv("VERIFIA_GPT_NAME", "gpt-4o-mini")
GPT_MODEL_TEMP = int(os.getenv("VERIFIA_GPT_TEMPERATURE", "0"))
GPT_MODEL = ChatOpenAI(temperature=GPT_MODEL_TEMP, model_name=GPT_MODEL_NAME)
MAX_RETRIES = int(os.getenv("VERIFIA_VALIDATOR_MAX_RETRIES", "3"))
