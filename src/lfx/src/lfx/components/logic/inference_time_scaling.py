import random
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict

from lfx.base.models.model import LCModelComponent
from lfx.field_typing import LanguageModel
from lfx.io import BoolInput, DropdownInput, HandleInput, IntInput, MessageInput, MultilineInput
from lfx.schema.message import Message
from lfx.template.field.base import Output


class InferenceTimeScalingWrapper(BaseChatModel):
    """Wrapper that applies inference-time scaling to any language model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_model: Any  # Use Any to avoid Pydantic validation issues
    budget: int = 3

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate multiple responses and select one randomly."""
        responses = []
        for _ in range(self.budget):
            response = self.base_model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            responses.append(response)

        # Return a random response
        return random.choice(responses)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Async version: generate multiple responses and select one randomly."""
        responses = []
        for _ in range(self.budget):
            response = await self.base_model._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            responses.append(response)

        # Return a random response
        return random.choice(responses)

    def bind_tools(self, tools, **kwargs):
        """Delegate tool binding to the base model."""
        # Return a new wrapper with the base model that has tools bound
        bound_base = self.base_model.bind_tools(tools, **kwargs)
        return InferenceTimeScalingWrapper(base_model=bound_base, budget=self.budget)

    @property
    def _llm_type(self) -> str:
        """Return the type of language model."""
        return f"inference_time_scaling_{self.base_model._llm_type}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters."""
        return {
            "base_model": self.base_model._identifying_params,
            "budget": self.budget,
        }


class InferenceTimeScalingComponent(LCModelComponent):
    display_name = "Inference-Time Scaling"
    description = "Applies inference-time scaling techniques to improve language model reasoning."
    name = "InferenceTimeScaling"
    icon = "timer"

    inputs = [
        HandleInput(
            name="language_model",
            display_name="Language Model",
            input_types=["LanguageModel"],
            info="The language model to use for generation.",
            required=True,
        ),
        MessageInput(
            name="input_value",
            display_name="Input",
            info="The input text to send to the model",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="A system message that helps set the behavior of the assistant",
            advanced=False,
        ),
        DropdownInput(
            name="algorithm",
            display_name="Algorithm",
            options=["Self-Consistency"],
            value="Self-Consistency",
            info="The inference-time scaling algorithm to use.",
            advanced=False,
        ),
        IntInput(
            name="budget",
            display_name="Budget",
            info="Number of times to call the model. A random response from all calls will be returned.",
            value=3,
            advanced=False,
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Model Response", name="text_output", method="text_response"),
        Output(display_name="Language Model", name="model_output", method="build_model"),
    ]

    def build_model(self) -> LanguageModel:
        """Return the language model wrapped with inference-time scaling.

        Returns:
            The wrapped language model that applies inference-time scaling.
        """
        budget = max(1, int(self.budget))

        # Wrap the language model with inference-time scaling
        wrapped_model = InferenceTimeScalingWrapper(
            base_model=self.language_model,
            budget=budget,
        )

        return wrapped_model

    async def text_response(self) -> Message:
        """Generate a text response using the language model with inference-time scaling.

        Calls the model multiple times based on the budget parameter and returns
        a randomly selected response from all generations.

        Returns:
            Message object containing a randomly selected model response.
        """
        model = self.build_model()
        budget = max(1, int(self.budget))  # Ensure budget is at least 1

        # Generate multiple responses
        responses = []
        for i in range(budget):
            result = await self.get_chat_result(
                runnable=model,
                stream=False,  # Disable streaming for multiple calls
                input_value=self.input_value,
                system_message=self.system_message,
            )
            responses.append(result)
            self.status = f"Generated {i + 1}/{budget} responses"

        # Select a random response
        selected_response = random.choice(responses)
        self.status = f"Selected 1 response from {budget} generations"

        return selected_response
