import requests
from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, MultilineInput, Output
from lfx.schema.data import Data

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
        DataInput(
            name="ground_truth_data",
            display_name="Ground Truth Data",
            info="The ground truth data.",
            input_types=["Data"],
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

    def evaluate_conversation(self) -> Data:
        """Evaluate the conversation history and return the score."""
        conversation_history = self.conversation_history
        ground_truth_data = self.ground_truth_data
        actual_hash_str = self.get_actual_hash_str()
        ground_truth_hash = ground_truth_data.data[ground_truth_data.text_key]
        reward = 1 if actual_hash_str == ground_truth_hash else 0
        result_data = {
            "reward": reward,
            "conversation_history": conversation_history,
            "ground_truth_hash": ground_truth_hash,
            "actual_hash_str": actual_hash_str
        }
        return Data(data=result_data)