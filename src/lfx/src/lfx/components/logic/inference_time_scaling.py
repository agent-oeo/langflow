import asyncio
import random
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import ConfigDict

from lfx.base.models.model import LCModelComponent
from lfx.components.logic.best_of_n import BestOfN
from lfx.field_typing import LanguageModel
from lfx.io import BoolInput, DropdownInput, HandleInput, IntInput, MessageInput, MultilineInput
from lfx.schema.dotdict import dotdict
from lfx.schema.message import Message
from lfx.template.field.base import Output
from lfx.utils.async_helpers import run_until_complete


class InferenceTimeScalingWrapper(BaseChatModel):
    """Wrapper that applies inference-time scaling to any language model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_model: Any  # Use Any to avoid Pydantic validation issues
    budget: int = 3
    algorithm: str = "Best-of-N"
    judge_llm: Any = None  # Only needed for Best-of-N
    judge_system_message: str | None = None  # Optional custom system message for judge
    top_n: int = 1  # Only needed for Best-of-N
    get_chat_result_fn: Any = None  # Function to call the model
    logger_fn: Any = None  # Function to log messages

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

    def invoke(self, input, config=None, **kwargs):
        """Invoke the model multiple times and select response based on algorithm."""
        return run_until_complete(self.ainvoke(input, config=config, **kwargs))

    async def ainvoke(self, input, config=None, **kwargs):
        """Async invoke: generate multiple responses and select based on algorithm."""
        if self.algorithm == "Best-of-N" and self.judge_llm and self.get_chat_result_fn:
            # Use Best-of-N algorithm
            print(f"\n🚀 Starting Best-of-N with budget={self.budget}, top_n={self.top_n}")

            best_of_n = BestOfN(
                judge_llm=self.judge_llm,
                get_chat_result_fn=self.get_chat_result_fn,
                logger_fn=self.logger_fn,
                judge_system_message=self.judge_system_message,
            )

            # Generate responses in parallel
            tasks = [self.base_model.ainvoke(input, config=config, **kwargs) for _ in range(self.budget)]
            responses = await asyncio.gather(*tasks)

            # Convert responses to Message objects if needed
            from langchain_core.messages import AIMessage
            import json

            messages = []
            print("Generated responses:")
            for resp in responses:
                print(resp, "\n")
                if isinstance(resp, AIMessage):
                    # Extract text content
                    text_content = resp.content if hasattr(resp, "content") else str(resp)

                    # If there are tool calls, include ONLY THE FIRST ONE in the text for judging
                    # We only want to judge the initial tool call decision, not subsequent calls
                    if hasattr(resp, "tool_calls") and resp.tool_calls:
                        tc = resp.tool_calls[0]  # Only take the first tool call
                        # tc can be a dict or a ToolCall object
                        if isinstance(tc, dict):
                            name = tc.get('name', 'unknown')
                            args = tc.get('args', {})
                        else:
                            name = getattr(tc, 'name', 'unknown')
                            args = getattr(tc, 'args', {})

                        tool_call_text = f"\n{name}({json.dumps(args, indent=2)})\n"
                        text_content = str(text_content) + tool_call_text

                    msg = Message(text=text_content)
                elif isinstance(resp, Message):
                    msg = resp
                else:
                    msg = Message(text=str(resp))
                messages.append(msg)

            # Score and select best response
            # Extract input text for scoring
            input_text = str(input)
            if hasattr(input, "content"):
                input_text = input.content
            elif isinstance(input, list) and len(input) > 0:
                # LangChain messages format
                input_text = str(input[-1])

            scores = await best_of_n._score_responses(messages, input_text, conversation_history=None)

            # Select top response
            scored_responses = list(enumerate(scores))
            scored_responses.sort(key=lambda x: x[1], reverse=True)
            selected_index = scored_responses[0][0]

            print(f"✅ Best-of-N completed. Selected best response from top {self.top_n} of {self.budget} generations\n")

            return responses[selected_index]
        else:
            # Self-Consistency: generate responses in parallel and select randomly
            print(f"\n🚀 Starting Self-Consistency with budget={self.budget}")

            responses = []
            for _ in range(self.budget):
                response = await self.base_model.ainvoke(input, config=config, **kwargs)
                responses.append(response)

            selected_idx = random.randint(0, len(responses) - 1)

            print(f"🎲 Randomly selected response {selected_idx + 1}/{len(responses)} (Self-Consistency)\n")

            return responses[selected_idx]

    def bind_tools(self, tools, **kwargs):
        """Delegate tool binding to the base model."""
        # Return a new wrapper with the base model that has tools bound
        bound_base = self.base_model.bind_tools(tools, **kwargs)
        return InferenceTimeScalingWrapper(
            base_model=bound_base,
            budget=self.budget,
            algorithm=self.algorithm,
            judge_llm=self.judge_llm,
            judge_system_message=self.judge_system_message,
            top_n=self.top_n,
            get_chat_result_fn=self.get_chat_result_fn,
            logger_fn=self.logger_fn,
        )

    def with_config(self, config=None, **kwargs):
        """Delegate config binding to the base model."""
        configured_base = self.base_model.with_config(config=config, **kwargs)
        return InferenceTimeScalingWrapper(
            base_model=configured_base,
            budget=self.budget,
            algorithm=self.algorithm,
            judge_llm=self.judge_llm,
            judge_system_message=self.judge_system_message,
            top_n=self.top_n,
            get_chat_result_fn=self.get_chat_result_fn,
            logger_fn=self.logger_fn,
        )

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
        MultilineInput(
            name="judge_system_message",
            display_name="Judge System Message",
            info="Custom system message for the judge LLM. Leave empty to use the default: 'You are an objective response evaluator. Provide only numeric scores in the requested format.'",
            advanced=True,
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
        top_n = max(1, int(getattr(self, "top_n", 1)))

        # Get custom judge system message if provided
        judge_sys_msg = getattr(self, "judge_system_message", None)
        if judge_sys_msg:
            judge_sys_msg = judge_sys_msg.strip()

        # Wrap the language model with inference-time scaling
        wrapped_model = InferenceTimeScalingWrapper(
            base_model=self.language_model,
            budget=budget,
            algorithm=self.algorithm,
            judge_llm=self.judge_llm if self.algorithm == "Best-of-N" else None,
            judge_system_message=judge_sys_msg if judge_sys_msg else None,
            top_n=top_n,
            get_chat_result_fn=self.get_chat_result,
            logger_fn=self.log,
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

            # Get conversation history if available
            conversation_history = None
            if hasattr(self, "graph") and hasattr(self.graph, "get_messages"):
                try:
                    conversation_history = self.graph.get_messages()
                except Exception:
                    pass  # If we can't get history, continue without it

            # Get custom judge system message if provided
            judge_sys_msg = getattr(self, "judge_system_message", None)
            if judge_sys_msg:
                judge_sys_msg = judge_sys_msg.strip()

            best_of_n = BestOfN(
                judge_llm=self.judge_llm,
                get_chat_result_fn=self.get_chat_result,
                logger_fn=None,  # No component logging, only print statements
                judge_system_message=judge_sys_msg if judge_sys_msg else None,
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

            self.status = f"Selected best response from {budget} generations (Best-of-N)"
        else:
            # Self-Consistency: generate responses in parallel and select randomly
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

            # Random selection
            selected_idx = random.randint(0, len(responses) - 1)
            selected_response = responses[selected_idx]

            self.status = f"Selected 1 response from {budget} generations (Self-Consistency)"

        return selected_response

    def update_build_config(self, build_config: dotdict, field_value: str, field_name: str | None = None) -> dotdict:
        """Update build configuration based on algorithm selection."""
        if field_name == "algorithm":
            # Show/hide Judge LLM, Judge System Message, and Top N based on algorithm
            if field_value == "Best-of-N":
                build_config["judge_llm"]["show"] = True
                build_config["judge_llm"]["required"] = True
                build_config["judge_system_message"]["show"] = True
                build_config["top_n"]["show"] = True
            else:
                build_config["judge_llm"]["show"] = False
                build_config["judge_llm"]["required"] = False
                build_config["judge_system_message"]["show"] = False
                build_config["top_n"]["show"] = False

        return build_config
