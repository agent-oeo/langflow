import random
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict

from lfx.base.models.model import LCModelComponent
from lfx.field_typing import LanguageModel
from lfx.io import BoolInput, DropdownInput, HandleInput, IntInput, MessageInput, MultilineInput
from lfx.schema.dotdict import dotdict
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
            options=["Self-Consistency", "Best-of-N"],
            value="Self-Consistency",
            info="The inference-time scaling algorithm to use.",
            real_time_refresh=True,
            advanced=False,
        ),
        IntInput(
            name="budget",
            display_name="Budget",
            info="Number of times to call the model. A random response from all calls will be returned.",
            value=3,
            advanced=False,
        ),
        HandleInput(
            name="judge_llm",
            display_name="Judge LLM",
            input_types=["LanguageModel"],
            info="The language model to use for judging responses (Best-of-N only).",
            required=False,
            advanced=False,
            show=False,  # Hidden by default
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
        a response based on the selected algorithm (Self-Consistency or Best-of-N).

        Returns:
            Message object containing the selected model response.
        """
        model = self.build_model()
        budget = max(1, int(self.budget))  # Ensure budget is at least 1
        algorithm = self.algorithm

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

        # Select response based on algorithm
        if algorithm == "Best-of-N" and self.judge_llm:
            selected_response = await self._select_best_of_n(responses)
        else:
            # Self-Consistency: random selection
            selected_response = random.choice(responses)
            self.status = f"Selected 1 response from {budget} generations (Self-Consistency)"

        return selected_response

    async def _select_best_of_n(self, responses: list[Message]) -> Message:
        """Use a judge LLM to select the best response from multiple candidates.

        Args:
            responses: List of Message objects to judge

        Returns:
            The best Message according to the judge LLM
        """
        if len(responses) == 1:
            return responses[0]

        # Create evaluation prompt
        candidates_text = "\n\n".join(
            [f"Response {i + 1}:\n{resp.text}" for i, resp in enumerate(responses)]
        )

        judge_prompt = f"""You are an expert evaluator. Your task is to select the best response from the following candidates based on quality, accuracy, and helpfulness.

Original Question: {self.input_value}

{candidates_text}

Analyze each response and select the best one. Reply with only the number (1, 2, 3, etc.) of the best response."""

        # Query judge LLM
        judge_result = await self.get_chat_result(
            runnable=self.judge_llm,
            stream=False,
            input_value=judge_prompt,
            system_message="You are a helpful judge that evaluates responses objectively.",
        )

        # Parse the judge's selection
        judge_text = judge_result.text.strip()
        try:
            # Try to extract a number from the judge's response
            import re

            numbers = re.findall(r"\d+", judge_text)
            if numbers:
                selected_idx = int(numbers[0]) - 1
                if 0 <= selected_idx < len(responses):
                    self.status = f"Selected response {selected_idx + 1}/{len(responses)} via Best-of-N judge"
                    return responses[selected_idx]
        except (ValueError, IndexError):
            pass

        # Fallback to first response if parsing fails
        self.status = f"Judge selection failed, using first response from {len(responses)} generations"
        return responses[0]

    def update_build_config(self, build_config: dotdict, field_value: str, field_name: str | None = None) -> dotdict:
        """Update build configuration based on algorithm selection."""
        if field_name == "algorithm":
            # Show/hide Judge LLM based on algorithm
            if field_value == "Best-of-N":
                build_config["judge_llm"]["show"] = True
                build_config["judge_llm"]["required"] = True
            else:
                build_config["judge_llm"]["show"] = False
                build_config["judge_llm"]["required"] = False

        return build_config
