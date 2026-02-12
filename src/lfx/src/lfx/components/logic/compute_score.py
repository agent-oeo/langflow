import requests
from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, MessageInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message

class ComputeScoreComponent(Component):
    """Compute the score of a conversation history and the ground truth hash string."""

    display_name: str = "Compute Score"
    description: str = "Compute the score of a conversation history and the ground truth hash string."
    icon = "Score"
    name = "ComputeScore"
    beta = True

    # I need conversation history and the ground truth hash str
    inputs = [
        DataInput(
            name="conversation_history",
            display_name="Conversation History",
            info="The conversation history between the user and the assistant.",
            input_types=["Data"],
            required=True,
        ),
        MessageInput(
            name="ground_truth_message",
            display_name="Ground Truth Message",
            info="The ground truth text.",
            input_types=["Message"],
            required=True,
        ),
    ]

    # output Data with tuple of (conversation_history, 1 or 0)
    outputs = [
        Output(name="evaluation_result", display_name="Evaluation Result", method="evaluate_conversation"),
    ]

    def get_actual_hash_str(self) -> str:
        response = requests.get("http://localhost:8001/hash")
        hash_data = response.json()
        hash_str = hash_data["hash"]
        return hash_str

    def _format_conversation_history(self, conversation_history: list[Message]) -> str:
        for msg in conversation_history:
            print(msg.text)
            print(msg.sender)


    def get_database_hash(self, intermediate_steps: list[Message]) -> str:
        """Compute the current hash of the in-memory airline database."""
        import json
        from tau_bench.envs.base import consistent_hash, to_hashable
        from tau_bench.envs.airline.data import load_data
        from tau_bench.envs.airline.tools import ALL_TOOLS
        data = load_data()

        tools = ALL_TOOLS
        terminate_tools = ["transfer_to_human_agent"]
        tools_map = {tool.get_info()["function"]["name"]: tool for tool in tools}
        for step in intermediate_steps:
            action = json.loads(step.text)
            if action['tool'] not in terminate_tools and action['tool'] in tools_map:
                _ = tools_map[action['tool']].invoke(data=data, **action['tool_input'])
        return consistent_hash(to_hashable(data)).strip()


    def evaluate_conversation(self) -> Data:
        """Evaluate the conversation history and return the score."""
        print('evaluate_conversation: Starting evaluation')
        conversation_history = self.conversation_history.data["conversation"]
        intermediate_steps = self.conversation_history.data["intermediate_steps"]
        database_hash = self.get_database_hash(intermediate_steps)
        print('database_hash:', database_hash)
        ground_truth_hash = self.ground_truth_message.text.strip()
        reward = 1 if database_hash == ground_truth_hash else 0
        result_data = {
            "reward": reward,
            "conversation_history": conversation_history,
            "intermediate_steps": intermediate_steps,
            "ground_truth_hash": ground_truth_hash,
            "database_hash": database_hash,
            "assistant_usage_metadata": self.conversation_history.data['assistant_usage_metadata'],
        }
        return Data(data=result_data)