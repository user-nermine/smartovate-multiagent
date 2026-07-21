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

# The Planificateur agent: breaks a complex request into clear sub-tasks
planner = AssistantAgent(
    name="planificateur",
    model_client=model_client,
    system_message=(
        "You are a project planner agent. "
        "When given a user request, break it down into a clear, numbered list of concrete sub-tasks "
        "that a coder agent and a reviewer agent could follow, in order. "
        "Keep the plan concise (3-6 steps). "
        "Do not write any code yourself -- only produce the plan. "
        "Do not say TERMINATE, another agent will decide when the task is done."
    ),
)

async def main():
    try:
        result = await planner.run(
            task="Build a Python function that takes a list of numbers and returns the average, with basic input validation."
        )
        for m in result.messages:
            print(f"--- {m.source} ---")
            print(m.content)
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())