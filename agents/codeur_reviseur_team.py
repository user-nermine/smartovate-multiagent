import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv()

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-10-21",
)

coder = AssistantAgent(
    name="codeur",
    model_client=model_client,
    system_message=(
        "You are a Python coding agent. "
        "Given a task or reviewer feedback, write or revise Python code to fulfill it. "
        "Always return the FULL corrected code in a single ```python code block. "
        "Do not include explanations outside the code block. "
        "Do not say TERMINATE or APPROVED -- only the reviewer decides that."
    ),
)

reviewer = AssistantAgent(
    name="reviseur",
    model_client=model_client,
    system_message=(
        "You are a code reviewer agent. "
        "Review the code from the codeur agent for correctness, edge cases, and clarity. "
        "If the code is correct and complete, reply with exactly: APPROVED TERMINATE "
        "If there are issues, give clear, concise, actionable feedback for the coder to fix -- "
        "keep feedback to the 2-3 most important issues only, do not nitpick endlessly. "
        "Do not rewrite the code yourself."
    ),
)

# Bug 1 fix (cahier des charges): prevent infinite loops with a hard cap on messages,
# in addition to the natural termination via the reviewer's "TERMINATE".
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(8)

team = RoundRobinGroupChat(
    [coder, reviewer],
    termination_condition=termination,
)

async def main():
    try:
        await Console(
            team.run_stream(
                task=(
                    "Write a Python function average(numbers: list) -> float that computes the average "
                    "of a list of numbers, with basic input validation (raise TypeError for non-list or "
                    "non-numeric input, ValueError for empty list)."
                )
            )
        )
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())