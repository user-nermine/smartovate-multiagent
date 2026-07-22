import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv()

st.set_page_config(page_title="Smartovate Multi-Agent System", layout="wide")
st.title("🤖 Smartovate — Système Multi-Agents")
st.caption("Planificateur → Codeur → Réviseur, propulsé par AutoGen + Azure OpenAI")

# --- User input ---
user_task = st.text_area(
    "Décrivez votre demande :",
    placeholder="Ex : Écris une fonction Python qui calcule la moyenne d'une liste de nombres, avec validation des entrées.",
    height=100,
)

run_button = st.button("🚀 Lancer les agents", type="primary")

# --- Agent setup (created fresh each run to avoid stale connections) ---
def build_team():
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
            "You are a project planner agent. Break the user's request into a clear, "
            "numbered list of 3-5 concrete sub-tasks for a coder agent. "
            "Do not write code. Do not say TERMINATE or APPROVED."
        ),
    )

    codeur = AssistantAgent(
        name="codeur",
        model_client=model_client,
        system_message=(
            "You are a Python coding agent. Follow the plan or reviewer feedback to write "
            "or revise Python code. Return the FULL code in a single ```python code block, "
            "no explanations outside it. Do not say TERMINATE or APPROVED."
        ),
    )

    reviseur = AssistantAgent(
        name="reviseur",
        model_client=model_client,
        system_message=(
            "You are a pragmatic code reviewer for a proof-of-concept. Check ONLY the core "
            "requirements explicitly requested. Do not ask for extra libraries, edge cases, "
            "or optimizations beyond what was asked. "
            "If core requirements are met, reply with exactly: APPROVED TERMINATE "
            "Otherwise give at most 1-2 critical, blocking issues only. Do not rewrite the code."
        ),
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
    team = RoundRobinGroupChat([planificateur, codeur, reviseur], termination_condition=termination)
    return team, model_client


async def run_agents(task: str):
    team, model_client = build_team()
    try:
        result = await team.run(task=task)
        return result.messages
    finally:
        await model_client.close()


import re

async def stream_agents(task: str, placeholder_container):
    """Runs the team and yields each message as it arrives, updating the UI live."""
    team, model_client = build_team()
    all_messages = []
    try:
        async for event in team.run_stream(task=task):
            if hasattr(event, "content") and hasattr(event, "source"):
                all_messages.append(event)
                with placeholder_container.container():
                    for m in all_messages:
                        with st.chat_message("user" if m.source == "user" else "assistant"):
                            st.markdown(f"**{m.source}**")
                            st.markdown(m.content)
    finally:
        await model_client.close()
    return all_messages


def extract_final_code(messages):
    """Grabs the last ```python ... ``` code block from the codeur's messages."""
    code_blocks = []
    for m in messages:
        if hasattr(m, "content") and getattr(m, "source", "") == "codeur":
            matches = re.findall(r"```python\s*(.*?)```", m.content, re.DOTALL)
            code_blocks.extend(matches)
    return code_blocks[-1].strip() if code_blocks else None


# --- Run and display ---
if run_button:
    if not user_task.strip():
        st.warning("Merci d'entrer une demande avant de lancer les agents.")
    else:
        st.subheader("Conversation en direct")
        live_area = st.empty()

        with st.spinner("Les agents travaillent..."):
            messages = asyncio.run(stream_agents(user_task, live_area))

        st.success("Conversation terminée !")

        final_code = extract_final_code(messages)
        if final_code:
            st.subheader("📄 Code final")
            st.code(final_code, language="python")
            st.download_button(
                label="⬇️ Exporter le code (.py)",
                data=final_code,
                file_name="generated_code.py",
                mime="text/x-python",
            )
        else:
            st.info("Aucun bloc de code final n'a été détecté dans la conversation.")