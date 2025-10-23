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
            input_types=["Message"],
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



    def evaluate_conversation(self) -> Data:
        """Evaluate the conversation history and return the score."""
        print('evaluate_conversation: Starting evaluation')
        conversation_history = self.conversation_history
        print('formatted conversation history:', self._format_conversation_history(conversation_history))
        ground_truth_hash = self.ground_truth_message.text
        actual_hash_str = self.get_actual_hash_str()
        reward = 1 if actual_hash_str == ground_truth_hash else 0
        result_data = {
            "reward": reward,
            "conversation_history": conversation_history,
            "ground_truth_hash": ground_truth_hash,
            "actual_hash_str": actual_hash_str
        }
        ret = Data(data=result_data)
        print('evaluate_conversation: Result data:', ret)
        return Data(data=result_data)