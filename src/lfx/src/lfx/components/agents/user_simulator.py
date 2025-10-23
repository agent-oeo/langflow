from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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
        MultilineInput(
            name="assistant_system_prompt",
            display_name="Assistant System Prompt",
            required=False,
            info="Optional system prompt for the assistant agent.",
        ),
        IntInput(
            name="max_turns",
            display_name="Assistant Turns",
            value=3,
            info="Maximum number of assistant responses to generate after the greeting.",
            range_spec=RangeSpec(min=1, max=50, step_type="int"),
        ),
    ]

    outputs = [
        Output(display_name="Conversation", name="conversation", method="conversation"),
    ]

    def _convert_lfx_message_to_lc_message(self, message: Message) -> BaseMessage:
        if message.sender == MESSAGE_SENDER_USER:
            return HumanMessage(content=message.text)
        else:
            return AIMessage(content=message.text)

    async def call_user_agent(self, conversation: list[Message]) -> Message:
        # swap sender from user to assistant and vice versa for messages in the conversation
        swapped_conversation: list[Message] = []
        for message in conversation:
            if message.sender == MESSAGE_SENDER_USER:
                swapped_conversation.append(Message(text=message.text, sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI))
            else:
                swapped_conversation.append(Message(text=message.text, sender=MESSAGE_SENDER_USER, sender_name=MESSAGE_SENDER_NAME_USER))
        
        # call the user agent with the swapped conversation
        # user agent is AgentExecutor
        args: dict[str, Any] = {
            "system_prompt": self.task_scenario,
            "chat_history": [self._convert_lfx_message_to_lc_message(message) for message in swapped_conversation[:-1]],
            "input": swapped_conversation[-1].text,
        }
        result = await self.user_agent.ainvoke(args)
        return Message(text=result["output"], sender=MESSAGE_SENDER_USER, sender_name=MESSAGE_SENDER_NAME_USER)

    async def call_assistant_agent(self, conversation: list[Message]) -> Message:
        # call assistant agent with the conversation
        args: dict[str, Any] = {
            "system_prompt": self.assistant_system_prompt,
            "chat_history": [self._convert_lfx_message_to_lc_message(message) for message in conversation[:-1]],
            "input": conversation[-1].text,
        }
        result = await self.assistant_agent.ainvoke(args)
        return Message(text=result["output"], sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)

    async def conversation(self) -> list[Message]:
        # get first user message
        user_message = await self.call_user_agent([Message(text="Hi! How can I help you today?", sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)])
        self.conversation = [user_message]
        for i in range(self.max_turns):
            assistant_message = await self.call_assistant_agent(self.conversation)
            self.conversation.append(assistant_message)
            user_message = await self.call_user_agent(self.conversation)
            self.conversation.append(user_message)

        return self.conversation