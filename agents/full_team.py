import os
import asyncio
import json
from datetime import datetime
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

planificateur = AssistantAgent(
    name="planificateur",
    model_client=model_client,
    system_message=(
        "You are a project planner agent. "
        "When given a user request, break it down into a clear, numbered list of concrete sub-tasks "
        "for a coder agent to follow. Keep the plan concise (3-5 steps). "
        "Do not write code yourself. Do not say TERMINATE or APPROVED."
    ),
)

codeur = AssistantAgent(
    name="codeur",
    model_client=model_client,
    system_message=(
        "You are a Python coding agent. "
        "Follow the planificateur's plan (or the reviseur's feedback) to write or revise Python code. "
        "Always return the FULL code in a single ```python code block, no explanations outside it. "
        "Do not say TERMINATE or APPROVED -- only the reviseur decides that."
    ),
)

reviseur = AssistantAgent(
    name="reviseur",
    model_client=model_client,
    system_message=(
        "You are a pragmatic code reviewer agent for a proof-of-concept project. "
        "Review the codeur's code for correctness on the CORE requirements only: "
        "does it handle the basic cases and the specific errors requested? "
        "Do NOT ask for additional features, edge cases, or libraries beyond what was explicitly requested "
        "(e.g. do not ask for numpy, Decimal, Fraction, or performance optimizations unless the task asked for them). "
        "If the core requirements are met, reply with exactly: APPROVED TERMINATE "
        "Otherwise, give at most 1-2 critical, blocking issues only. "
        "Do not rewrite the code yourself."
    ),

)

# Speaker selection method: Round-robin (fixed order: planificateur -> codeur -> reviseur -> codeur -> reviseur ...)
# Chosen for predictability in this PoC -- each role always gets a turn in the same sequence.
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)

team = RoundRobinGroupChat(
    [planificateur, codeur, reviseur],
    termination_condition=termination,
)

async def save_log(messages, task):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/conversation_{timestamp}.json"
    log_data = {
        "task": task,
        "timestamp": timestamp,
        "messages": [
            {"source": m.source, "content": m.content}
            for m in messages
            if hasattr(m, "content")
        ],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Conversation log saved to {log_path}")

async def main():
    task = (
        "Build a Python function that takes a list of numbers and returns the average, "
        "with basic input validation."
    )
    try:
        result = await team.run(task=task)
        # Print the conversation
        for m in result.messages:
            if hasattr(m, "content"):
                print(f"\n---------- {m.source} ----------")
                print(m.content)
        await save_log(result.messages, task)
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())