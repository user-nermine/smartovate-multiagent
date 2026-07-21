import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.ui import Console


load_dotenv()

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-10-21",
)

assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful assistant. When the task is complete, reply with TERMINATE.",
)

termination = TextMentionTermination("TERMINATE")

team = RoundRobinGroupChat(
    [assistant],
    termination_condition=termination,
    # no max_turns this time
)


async def main():
    try:
        await Console(
            team.run_stream(
                task="Say hello and tell me one interesting fact about Python in one sentence."
            )
        )
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())