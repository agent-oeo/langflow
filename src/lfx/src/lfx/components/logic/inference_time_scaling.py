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
        # Handle ChatPromptValue objects (they have a messages attribute)
        if hasattr(input, 'messages'):
            input = input.messages

        # Extract conversation history (0 to n-1) from input if it's a list
        conversation_history = []
        if isinstance(input, list) and len(input) > 1:
            conversation_history = input[:-1]

        # Define generator function that converts LangChain responses to Messages
        async def generate_one(input_val):
            from langchain_core.messages import AIMessage
            import json

            resp = await self.base_model.ainvoke(input_val, config=config, **kwargs)

            # Convert to Message with tool calls
            if isinstance(resp, AIMessage):
                text_content = resp.content if hasattr(resp, "content") else str(resp)

                # Include only the first tool call for judging
                if hasattr(resp, "tool_calls") and resp.tool_calls:
                    tc = resp.tool_calls[0]
                    name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                    args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                    tool_call_text = f"\n{name}({json.dumps(args, indent=2)})\n"
                    text_content = str(text_content) + tool_call_text

                return Message(text=text_content), resp
            elif isinstance(resp, Message):
                return resp, resp
            else:
                return Message(text=str(resp)), resp

        # Use shared ITS algorithm (import needed to avoid circular dependency)
        from lfx.components.logic.inference_time_scaling import InferenceTimeScalingComponent
        selected_response, _ = await InferenceTimeScalingComponent._run_its_algorithm(
            algorithm=self.algorithm,
            budget=self.budget,
            top_n=self.top_n,
            judge_llm=self.judge_llm,
            judge_system_message=self.judge_system_message,
            get_chat_result_fn=self.get_chat_result_fn,
            generate_fn=generate_one,
            input_data=input,
            conversation_history=conversation_history,
        )

        return selected_response

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

    @staticmethod
    async def _run_its_algorithm(
        algorithm: str,
        budget: int,
        top_n: int,
        judge_llm,
        judge_system_message: str | None,
        get_chat_result_fn,
        generate_fn,
        input_data,
        conversation_history=None,
    ):
        """Shared helper to run ITS algorithm (Best-of-N or Self-Consistency).

        Args:
            algorithm: "Best-of-N" or "Self-Consistency"
            budget: Number of responses to generate
            top_n: Top N responses to consider (Best-of-N only)
            judge_llm: Judge model (Best-of-N only)
            judge_system_message: Custom judge system message
            get_chat_result_fn: Function to call judge LLM
            generate_fn: Async function that generates a single response
            input_data: Input for generation (varies by context)
            conversation_history: Conversation history for Best-of-N judging

        Returns:
            Selected response and list of Message objects (for judging)
        """
        if algorithm == "Best-of-N" and judge_llm:
            print(f"\n🚀 Starting Best-of-N with budget={budget}, top_n={top_n}")

            best_of_n = BestOfN(
                judge_llm=judge_llm,
                get_chat_result_fn=get_chat_result_fn,
                logger_fn=None,
                judge_system_message=judge_system_message,
            )

            # Generate responses in parallel
            tasks = [generate_fn(input_data) for _ in range(budget)]
            results = await asyncio.gather(*tasks)

            # results is a list of (Message, original_response) tuples
            messages = [r[0] for r in results]
            responses = [r[1] for r in results]

            # Extract current message from input_data
            current_message = input_data
            if isinstance(input_data, list) and len(input_data) > 0:
                current_message = input_data[-1]

            # Score and select
            print("Messages:", messages)
            print("Current message:", current_message)
            print("Current message Type:", type(current_message))
            print("Conversation history:", conversation_history)
            scores = await best_of_n._score_responses(messages, current_message, conversation_history=conversation_history)
            scored_responses = list(enumerate(scores))
            scored_responses.sort(key=lambda x: x[1], reverse=True)
            selected_index = scored_responses[0][0]

            print(f"✅ Best-of-N completed. Selected best response from top {top_n} of {budget} generations\n")
            return responses[selected_index], messages
        else:
            # Self-Consistency
            print(f"\n🚀 Starting Self-Consistency with budget={budget}")

            results = []
            for _ in range(budget):
                result = await generate_fn(input_data)
                results.append(result)

            messages = [r[0] for r in results]
            responses = [r[1] for r in results]

            selected_idx = random.randint(0, len(responses) - 1)
            print(f"🎲 Randomly selected response {selected_idx + 1}/{len(responses)} (Self-Consistency)\n")

            return responses[selected_idx], messages

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

        # Define generator function for component context
        async def generate_one(_input_data):
            msg = await self.get_chat_result(
                runnable=self.language_model,
                stream=False,
                input_value=self.input_value,
                system_message=self.system_message,
            )
            return msg, msg  # Return (Message, Message) tuple

        # Use shared ITS algorithm
        self.status = f"Generating {budget} responses in parallel..."

        top_n = max(1, int(getattr(self, "top_n", 1)))
        selected_response, _ = await self._run_its_algorithm(
            algorithm=algorithm,
            budget=budget,
            top_n=top_n,
            judge_llm=self.judge_llm,
            judge_system_message=judge_sys_msg,
            get_chat_result_fn=self.get_chat_result,
            generate_fn=generate_one,
            input_data=self.input_value,
            conversation_history=conversation_history,
        )

        self.status = f"Selected best response from {budget} generations ({algorithm})"
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
