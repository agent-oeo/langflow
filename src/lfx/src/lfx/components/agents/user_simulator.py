from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor

from lfx.custom.custom_component.component import Component
from lfx.field_typing import RangeSpec
from lfx.io import HandleInput, IntInput, MultilineInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message
from lfx.utils.constants import (
    MESSAGE_SENDER_AI,
    MESSAGE_SENDER_NAME_AI,
    MESSAGE_SENDER_NAME_USER,
    MESSAGE_SENDER_USER,
)


class UserSimulatorComponent(Component):
    """Simulate a scripted user speaking with an assistant agent around a task scenario."""

    display_name: str = "User Simulator"
    description: str = (
        "Feed a task scenario to a user agent, seed the opening message, "
        "and facilitate a turn-based exchange with an assistant agent."
    )
    icon = "Users"
    name = "UserSimulator"
    beta = True

    inputs = [
        HandleInput(
            name="user_agent",
            display_name="User Agent",
            input_types=["Agent"],
            required=True,
            info="Agent that role-plays the user. Receives the task scenario as its system prompt.",
        ),
        HandleInput(
            name="assistant_agent",
            display_name="Assistant Agent",
            input_types=["Agent"],
            required=True,
            info="Agent that replies to the simulated user messages.",
        ),
        MultilineInput(
            name="task_scenario",
            display_name="Task Scenario",
            required=True,
            info="Describe the situation or objective guiding the user agent.",
        ),
        IntInput(
            name="max_turns",
            display_name="Assistant Turns",
            value=3,
            info="Maximum number of assistant responses to generate after the greeting.",
            range_spec=RangeSpec(min=1, max=10, step_type="int"),
        ),
    ]

    outputs = [
        Output(display_name="Conversation", name="conversation", method="conversation"),
        Output(display_name="Assistant Reply", name="assistant_reply", method="assistant_reply"),
        Output(display_name="Transcript Data", name="transcript_data", method="transcript_data", cache=True),
    ]

    async def conversation(self) -> list[Message]:
        return await self._ensure_conversation()

    async def assistant_reply(self) -> Message:
        conversation = await self._ensure_conversation()
        for message in reversed(conversation):
            if message.sender == MESSAGE_SENDER_AI:
                return message
        return Message(text="", sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)

    async def transcript_data(self) -> Data:
        conversation = await self._ensure_conversation()
        serialized = [message.model_dump() for message in conversation]
        return Data(data={"messages": serialized, "turns": len(conversation)})

    async def _ensure_conversation(self) -> list[Message]:
        if not hasattr(self, "_conversation_cache"):
            self._conversation_cache: list[Message] | None = None
        if self._conversation_cache is None:
            self._conversation_cache = await self._simulate_conversation()
        return self._conversation_cache

    async def _simulate_conversation(self) -> list[Message]:
        scenario = (self.task_scenario or "").strip()
        user_agent = self._coerce_agent(self.user_agent, "user_agent")
        assistant_agent = self._coerce_agent(self.assistant_agent, "assistant_agent")
        max_turns = max(1, int(self.max_turns))

        conversation: list[Message] = [self._bootstrap_greeting()]
        assistant_turns = 0

        while assistant_turns < max_turns:
            user_prompt = conversation[-1].text or ""
            assistant_reply = await self._invoke_agent(
                agent=assistant_agent,
                incoming=user_prompt,
                history=conversation[:-1],
                system_prompt=None,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )
            conversation.append(assistant_reply)
            assistant_turns += 1

            if not assistant_reply.text or not assistant_reply.text.strip():
                break
            if assistant_turns >= max_turns:
                break

            user_reply = await self._invoke_agent(
                agent=user_agent,
                incoming=assistant_reply.text or "",
                history=conversation[:-1],
                system_prompt=scenario,
                sender=MESSAGE_SENDER_USER,
                sender_name=MESSAGE_SENDER_NAME_USER,
            )
            conversation.append(user_reply)
            if not user_reply.text or not user_reply.text.strip():
                break

        self.status = f"Simulated {len(conversation)} messages ({assistant_turns} assistant turns)."
        return conversation

    @staticmethod
    def _bootstrap_greeting() -> Message:
        message = Message(
            text="Hi! How can I help you today?",
            sender=MESSAGE_SENDER_USER,
            sender_name=MESSAGE_SENDER_NAME_USER,
        )
        message.data.setdefault("role", "user")
        message.data.setdefault("source", "simulator")
        return message

    @staticmethod
    def _coerce_agent(agent: Any, field_name: str) -> Any:
        value = agent[0] if isinstance(agent, (list, tuple)) else agent
        if isinstance(value, AgentExecutor):
            return value
        if hasattr(value, "ainvoke") and hasattr(value, "input_keys"):
            return value
        msg = f"Expected '{field_name}' to be an AgentExecutor-like object with 'ainvoke'."
        raise TypeError(msg)

    async def _invoke_agent(
        self,
        *,
        agent: Any,
        incoming: str,
        history: list[Message],
        system_prompt: str | None,
        sender: str,
        sender_name: str,
    ) -> Message:
        payload, applied_system_prompt, primary_key = self._build_payload(agent, incoming, history, system_prompt)
        result = await agent.ainvoke(payload)
        text, raw = self._extract_text(result)
        message = Message(text=text or "", sender=sender, sender_name=sender_name)
        message.data["raw_output"] = raw
        history_entries = payload.get("chat_history") or payload.get("messages")
        message.data["metadata"] = {
            "input_key": primary_key,
            "history_length": len(history_entries) if isinstance(history_entries, list) else 0,
        }
        if applied_system_prompt:
            message.data["metadata"]["system_prompt"] = applied_system_prompt
        return message

    @staticmethod
    def _build_payload(
        agent: Any,
        incoming: str,
        history: list[Message],
        system_prompt: str | None,
    ) -> tuple[dict[str, Any], str | None, str]:
        input_keys = list(getattr(agent, "input_keys", []))
        if not input_keys:
            input_keys = ["input"]

        payload: dict[str, Any] = {}
        primary_key = "input" if "input" in input_keys else input_keys[0]
        payload[primary_key] = incoming

        if history:
            history_messages = [message.to_lc_message() for message in history if message.text]
            if history_messages:
                if "chat_history" in input_keys:
                    payload["chat_history"] = history_messages
                elif "messages" in input_keys:
                    payload["messages"] = history_messages

        applied_system_prompt: str | None = None
        if system_prompt and system_prompt.strip():
            if "system_prompt" in input_keys:
                payload["system_prompt"] = system_prompt
                applied_system_prompt = system_prompt
            elif primary_key == "input":
                payload[primary_key] = f"{system_prompt}\n\n{incoming}"

        return payload, applied_system_prompt, primary_key

    @staticmethod
    def _extract_text(result: Any) -> tuple[str, Any]:
        if isinstance(result, Message):
            return result.text or "", result
        if isinstance(result, Data):
            text_value = result.data.get(result.text_key, result.get_text())
            return (text_value or "") if isinstance(text_value, str) else str(text_value), result
        if isinstance(result, str):
            return result, result
        if isinstance(result, dict):
            for key in ("output", "response", "result", "text", "answer"):
                if key in result and result[key] is not None:
                    value = result[key]
                    if isinstance(value, (Message, Data)):
                        text, _ = UserSimulatorComponent._extract_text(value)
                        return text, result
                    return (value or "") if isinstance(value, str) else str(value), result
            return str(result), result
        return str(result), result
