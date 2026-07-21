import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv()

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-10-21",
)

# The Codeur agent: writes Python code based on instructions
coder = AssistantAgent(
    name="codeur",
    model_client=model_client,
    system_message=(
        "You are a Python coding agent. "
        "Given a plan or a task description, write clean, correct Python code that fulfills it. "
        "Always wrap your code in a single Python code block (```python ... ```). "
        "Include a short docstring. Do not include explanations outside the code block. "
        "If a reviewer gives you feedback, revise your code accordingly and return the full corrected version. "
        "Do not say TERMINATE, the reviewer will decide when the code is approved."
    ),
)

async def main():
    try:
        result = await coder.run(
            task=(
                "Write a Python function average(numbers: list) -> float that computes the average "
                "of a list of numbers. It should raise TypeError if the input is not a list or contains "
                "non-numeric values, and ValueError if the list is empty."
            )
        )
        for m in result.messages:
            print(f"--- {m.source} ---")
            print(m.content)
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())