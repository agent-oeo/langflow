from __future__ import annotations

from langchain_core.callbacks import UsageMetadataCallbackHandler
import requests
import json
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
        Output(display_name="Conversation", name="conversation", method="create_conversation"),
    ]

    def _convert_lfx_message_to_lc_message(self, message: Message) -> BaseMessage:
        print(f"Converting LFX message to LC message: {message.text[:50]}...")
        if message.sender == MESSAGE_SENDER_USER:
            print("Converting to HumanMessage")
            return HumanMessage(content=message.text)
        else:
            print("Converting to AIMessage")
            return AIMessage(content=message.text)

    async def call_user_agent(self, conversation: list[Message]) -> Message:
        print(f"Calling user agent with conversation of {len(conversation)} messages")
        # swap sender from user to assistant and vice versa for messages in the conversation
        swapped_conversation: list[Message] = []
        for message in conversation:
            print(f"Processing message from {message.sender}: {message.text[:30]}...")
            if message.sender == MESSAGE_SENDER_USER:
                print("Swapping USER to AI")
                swapped_conversation.append(Message(text=message.text, sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI))
            else:
                print("Swapping AI to USER")
                swapped_conversation.append(Message(text=message.text, sender=MESSAGE_SENDER_USER, sender_name=MESSAGE_SENDER_NAME_USER))
        
        print(f"Swapped conversation has {len(swapped_conversation)} messages")
        # call the user agent with the swapped conversation
        # user agent is AgentExecutor
        args: dict[str, Any] = {
            "system_prompt": self.task_scenario,
            "chat_history": [self._convert_lfx_message_to_lc_message(message) for message in swapped_conversation[:-1]],
            "input": swapped_conversation[-1].text,
        }
        print(f"Calling user agent with args: system_prompt length={len(self.task_scenario)}, chat_history length={len(args['chat_history'])}, input={args['input'][:50]}...")
        result = await self.user_agent.ainvoke(args)
        print(f"User agent result: {result['output'][:100]}...")
        return Message(text=result["output"], sender=MESSAGE_SENDER_USER, sender_name=MESSAGE_SENDER_NAME_USER)

    async def call_assistant_agent(self, conversation: list[Message]) -> Message:
        cb = UsageMetadataCallbackHandler()
        print(f"Calling assistant agent with conversation of {len(conversation)} messages")
        # call assistant agent with the conversation
        args: dict[str, Any] = {
            "system_prompt": self.assistant_system_prompt,
            "chat_history": [self._convert_lfx_message_to_lc_message(message) for message in conversation[:-1]],
            "input": conversation[-1].text,
        }
        config = {"callbacks": [cb]}
        print(f"Calling assistant agent with args: system_prompt length={len(self.assistant_system_prompt or '')}, chat_history length={len(args['chat_history'])}, input={args['input'][:50]}...")
        result = await self.assistant_agent.ainvoke(args, config=config)
        self.intermediate_steps.extend(result["intermediate_steps"])
        print('-'*50)
        print('INTERMEDIATE STEPS:', result["intermediate_steps"])
        print('-'*50)
        print(f"Assistant agent result: {result['output'][:100]}...")
        return {
            'message': Message(text=result["output"], sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI),
            'usage': list(cb.usage_metadata.values())[0] if len(cb.usage_metadata) else {}
        }

    def add_usage(self, usage_metadata, other):
        if usage_metadata is None: return other

        ret = {'input_token_details': {}}
        ret['input_tokens'] = usage_metadata.get('input_tokens', 0) + other.get('input_tokens', 0)
        ret['input_token_details']['cache_read'] = usage_metadata.get('input_token_details', {'cache_read': 0})['cache_read'] + other.get('input_token_details', {'cache_read': 0})['cache_read']
        ret['output_tokens'] = usage_metadata.get('output_tokens', 0) + other.get('output_tokens', 0)
        ret['total_tokens'] = usage_metadata.get('total_tokens', 0) + other.get('total_tokens', 0)
        return ret


    async def create_conversation(self) -> Data:
        print("Starting conversation simulation")
        usage_metadata = None
        # reset the env
        print("Resetting environment via HTTP request")
        try:
            response = requests.post("http://localhost:8001/reload")
            response.raise_for_status()
            print("Environment reset successful")
        except Exception as e:
            print(f"Failed to reset env: {e}")
        
        # get first user message
        print("Getting first user message")
        user_message = await self.call_user_agent([Message(text="Hi! How can I help you today?", sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)])
        print(f"First user message: {user_message.text[:100]}...")
        self.conversation = [user_message]
        self.intermediate_steps = []
        
        print(f"Starting conversation loop for {self.max_turns} turns")
        for i in range(self.max_turns):
            print(f"Turn {i+1}/{self.max_turns}")
            ret = await self.call_assistant_agent(self.conversation)
            assistant_message, _ = ret['message'], ret['usage']
            print(f"Assistant response: {assistant_message.text[:100]}...")
            self.conversation.append(assistant_message)
            usage_metadata = self.add_usage(usage_metadata, _)
            
            user_message = await self.call_user_agent(self.conversation)
            print(f"User response: {user_message.text[:100]}...")
            if user_message.text == "###STOP###":
                print("User sent stop signal, ending conversation")
                break
            self.conversation.append(user_message)

        print(f"Conversation completed with {len(self.conversation)} total messages")

        # map intermediate steps to json
        ret_steps = []
        for step in self.intermediate_steps:
            ret_steps.append(Message(text=json.dumps({
                "tool": step[0].tool,
                "tool_input": step[0].tool_input,
            }), sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI))

        return Data(data={"conversation": self.conversation, "intermediate_steps": ret_steps, "assistant_usage_metadata": usage_metadata})