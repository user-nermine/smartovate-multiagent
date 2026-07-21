import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load variables from .env
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-10-21",  # stable, working API version
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
    max_completion_tokens=300,
)

print("✅ Connection successful!")
print("Response:", response.choices[0].message.content)