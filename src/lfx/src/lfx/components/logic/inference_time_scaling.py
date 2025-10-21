import asyncio
import random
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict

from lfx.base.models.model import LCModelComponent
from lfx.components.logic.best_of_n import BestOfN
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
            options=["Best-of-N", "Self-Consistency"],
            value="Best-of-N",
            info="The inference-time scaling algorithm to use.",
            real_time_refresh=True,
            advanced=False,
        ),
        IntInput(
            name="budget",
            display_name="Budget",
            info="Number of responses to generate in parallel. The best response will be selected based on the algorithm.",
            value=3,
            advanced=False,
        ),
        IntInput(
            name="top_n",
            display_name="Top N",
            info="Number of top-scoring responses to consider (Best-of-N only). Default is 1 (best only).",
            value=1,
            advanced=True,
            show=False,  # Hidden by default, shown when Best-of-N is selected
        ),
        HandleInput(
            name="judge_llm",
            display_name="Judge LLM",
            input_types=["LanguageModel"],
            info="The language model to use for judging responses (Best-of-N only).",
            required=True,
            advanced=False,
            show=True,  # Shown by default since Best-of-N is default
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
        budget = max(1, int(self.budget))  # Ensure budget is at least 1
        algorithm = self.algorithm

        # Select response based on algorithm
        if algorithm == "Best-of-N" and self.judge_llm:
            # Use BestOfN algorithm with parallel async generation
            top_n = max(1, int(getattr(self, "top_n", 1)))  # Default to 1 if not set
            self.log(f"🚀 Starting Best-of-N with budget={budget}, top_n={top_n}")

            # Get conversation history if available
            conversation_history = None
            if hasattr(self, "graph") and hasattr(self.graph, "get_messages"):
                try:
                    conversation_history = self.graph.get_messages()
                except Exception:
                    pass  # If we can't get history, continue without it

            best_of_n = BestOfN(
                judge_llm=self.judge_llm,
                get_chat_result_fn=self.get_chat_result,
                logger_fn=self.log,  # Pass the log function for detailed logging
            )

            self.status = f"Generating {budget} responses in parallel..."

            selected_response = await best_of_n.ainfer(
                lm=self.language_model,
                input_value=self.input_value,
                system_message=self.system_message,
                budget=budget,
                top_n=top_n,
                conversation_history=conversation_history,
                return_response_only=True,
            )

            self.log(f"✅ Best-of-N completed. Selected best response from top {top_n} of {budget} generations")
            self.status = f"Selected best response from {budget} generations (Best-of-N)"
        else:
            # Self-Consistency: generate responses in parallel and select randomly
            self.log(f"🚀 Starting Self-Consistency with budget={budget}")
            self.status = f"Generating {budget} responses in parallel..."

            tasks = [
                self.get_chat_result(
                    runnable=self.language_model,
                    stream=False,
                    input_value=self.input_value,
                    system_message=self.system_message,
                )
                for _ in range(budget)
            ]

            responses = await asyncio.gather(*tasks)

            # Log all generated responses
            self.log(f"✅ Generated {len(responses)} responses")
            for i, response in enumerate(responses, 1):
                preview = response.text[:200] + "..." if len(response.text) > 200 else response.text
                self.log(f"Response {i}: {preview}")

            # Random selection
            selected_idx = random.randint(0, len(responses) - 1)
            selected_response = responses[selected_idx]

            self.log(f"🎲 Randomly selected response {selected_idx + 1}/{len(responses)} (Self-Consistency)")
            self.status = f"Selected 1 response from {budget} generations (Self-Consistency)"

        return selected_response


    def update_build_config(self, build_config: dotdict, field_value: str, field_name: str | None = None) -> dotdict:
        """Update build configuration based on algorithm selection."""
        if field_name == "algorithm":
            # Show/hide Judge LLM and Top N based on algorithm
            if field_value == "Best-of-N":
                build_config["judge_llm"]["show"] = True
                build_config["judge_llm"]["required"] = True
                build_config["top_n"]["show"] = True
            else:
                build_config["judge_llm"]["show"] = False
                build_config["judge_llm"]["required"] = False
                build_config["top_n"]["show"] = False

        return build_config
