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

# The Réviseur agent: reviews code from the Codeur and either approves or requests changes
reviewer = AssistantAgent(
    name="reviseur",
    model_client=model_client,
    system_message=(
        "You are a code reviewer agent. "
        "You will be given Python code written by a coder agent. "
        "Check it for correctness, edge cases, and clarity. "
        "If the code is correct and complete, reply with 'APPROVED' followed by TERMINATE. "
        "If there are issues, list them clearly and concisely as actionable feedback for the coder to fix. "
        "Do not rewrite the code yourself -- only point out what needs to change. "
        "Only say TERMINATE when you approve the code."
    ),
)

async def main():
    code_to_review = '''
```python
from numbers import Real
from typing import List

def average(numbers: List) -> float:
    """
    Compute the average of a list of numeric values.
    """
    if not isinstance(numbers, list):
        raise TypeError("Input must be a list.")
    if len(numbers) == 0:
        raise ValueError("List is empty.")
    total = 0.0
    count = 0
    for idx, item in enumerate(numbers):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"List contains non-numeric value at index {idx}: {item!r}")
        total += float(item)
        count += 1
    return total / count
```
'''
    try:
        result = await reviewer.run(task=f"Please review this code:\n{code_to_review}")
        for m in result.messages:
            print(f"--- {m.source} ---")
            print(m.content)
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())