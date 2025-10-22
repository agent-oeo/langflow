"""Best-of-N inference-time scaling algorithm implementation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lfx.schema.message import Message

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def format_message_with_tool_calls(msg: Message) -> str:
    """Format a message including any tool calls for display/judging.

    Args:
        msg: Message object that may contain tool call information

    Returns:
        Formatted string with text and tool calls
    """
    text = msg.text or ""

    # Check if message has tool call information in properties or content_blocks
    # This is a best-effort attempt to extract tool calls from Langflow Messages
    formatted = str(text)

    # TODO: Once Langflow Message schema supports tool_calls natively, extract them here
    # For now, we rely on the text content which should include tool calls if properly converted

    return formatted


@dataclass
class BestOfNResult:
    """Result from Best-of-N algorithm containing responses, scores, and selection."""

    responses: list[Message]
    scores: list[float]
    selected_index: int

    @property
    def the_one(self) -> Message:
        """Return the best selected response."""
        return self.responses[self.selected_index]


class BestOfN:
    """Best-of-N inference-time scaling algorithm.

    Generates multiple responses in parallel and uses a judge LLM to score
    and select the best response.
    """

    def __init__(
        self,
        judge_llm: BaseChatModel,
        get_chat_result_fn: Any,
        logger_fn: Any = None,
        judge_system_message: str | None = None,
    ) -> None:
        """Initialize Best-of-N algorithm.

        Args:
            judge_llm: Language model to use for judging/scoring responses
            get_chat_result_fn: Async function to get chat results from a model
            logger_fn: Optional function to log messages to component logs
            judge_system_message: Optional custom system message for the judge. If None, uses default.
        """
        self.judge_llm = judge_llm
        self.get_chat_result_fn = get_chat_result_fn
        self.logger_fn = logger_fn
        self.judge_system_message = (
            judge_system_message
            if judge_system_message
            else "You are an objective response evaluator. Provide only numeric scores in the requested format."
        )

    async def ainfer(
        self,
        lm: BaseChatModel,
        input_value: str | Message,
        system_message: str | None,
        budget: int,
        top_n: int = 1,
        conversation_history: list[Message] | None = None,
        return_response_only: bool = True,
    ) -> Message | BestOfNResult:
        """Run inference asynchronously with best-of-n.

        Args:
            lm: Language model to generate responses
            input_value: Input prompt or message
            system_message: System message to prepend
            budget: Number of responses to generate
            top_n: Number of top responses to return (default 1 for best only)
            conversation_history: Optional conversation history to provide context to judge
            return_response_only: If True, return only the best response. If False, return full result.

        Returns:
            The best response (if return_response_only=True) or BestOfNResult object
        """
        # Generate responses in parallel
        tasks = [
            self.get_chat_result_fn(
                runnable=lm,
                stream=False,
                input_value=input_value,
                system_message=system_message,
            )
            for _ in range(budget)
        ]

        responses = await asyncio.gather(*tasks)

        # Score responses using judge LLM (all at once in a single prompt)
        scores = await self._score_responses(responses, input_value, conversation_history)

        # Select top N responses by score
        scored_responses = list(enumerate(scores))
        scored_responses.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored_responses[:top_n]]

        # Primary selection is the best one
        selected_index = top_indices[0]

        # Return the result
        result = BestOfNResult(
            responses=responses,
            scores=scores,
            selected_index=selected_index,
        )

        return result.the_one if return_response_only else result

    async def _score_responses(
        self,
        responses: list[Message],
        original_input: str | Message,
        conversation_history: list[Message] | None = None,
    ) -> list[float]:
        """Score responses using the judge LLM.

        Evaluates all responses in a single judge prompt for consistency.

        Args:
            responses: List of responses to score
            original_input: Original input/question
            conversation_history: Optional conversation history for context

        Returns:
            List of scores (one per response)
        """
        # Extract text from original input
        if isinstance(original_input, Message):
            input_text = original_input.text
        elif hasattr(original_input, 'content'):
            # LangChain message - extract content
            content = original_input.content
            # If content is a list (multi-modal), extract text
            if isinstance(content, list):
                content = ' '.join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in content])
            input_text = str(content)

            # If AI message with tool calls, append tool call info
            if hasattr(original_input, 'tool_calls') and original_input.tool_calls:
                import json
                tool_calls_text = "\n[Tool Calls]:\n"
                for tc in original_input.tool_calls:
                    tc_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                    tc_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                    tool_calls_text += f"  - {tc_name}({json.dumps(tc_args)})\n"
                input_text += tool_calls_text
        else:
            input_text = str(original_input)

        # Build conversation history context
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            history_context = "\n## Conversation History:\n"
            for msg in conversation_history:
                # Handle both LangChain messages (content) and Langflow messages (text)
                if hasattr(msg, 'content'):
                    # LangChain message - extract content
                    content = msg.content
                    # If content is a list (multi-modal), extract text
                    if isinstance(content, list):
                        content = ' '.join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in content])
                    msg_text = str(content)

                    # Get role from message type
                    msg_type = type(msg).__name__
                    if 'System' in msg_type:
                        sender = "System"
                    elif 'Human' in msg_type or 'User' in msg_type:
                        sender = "User"
                    elif 'Tool' in msg_type:
                        sender = "Tool"
                        # For tool messages, also include tool name if available
                        if hasattr(msg, 'name') and msg.name:
                            sender = f"Tool ({msg.name})"
                    elif 'AI' in msg_type or 'Assistant' in msg_type:
                        sender = "Assistant"
                        # For AI messages with tool calls, append tool call info
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            import json
                            tool_calls_text = "\n[Tool Calls]:\n"
                            for tc in msg.tool_calls:
                                tc_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                                tc_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                                tool_calls_text += f"  - {tc_name}({json.dumps(tc_args)})\n"
                            msg_text += tool_calls_text
                    else:
                        sender = "Unknown"
                elif hasattr(msg, 'text'):
                    # Langflow Message object
                    msg_text = msg.text
                    sender = getattr(msg, "sender", "Unknown")
                else:
                    msg_text = str(msg)
                    sender = "Unknown"

                history_context += f"**{sender}**: {msg_text}\n"
            history_context += "\n"
    
        # Build the judge prompt with all responses
        responses_text = ""
        for i, response in enumerate(responses, 1):
            formatted_resp = format_message_with_tool_calls(response)
            responses_text += f"\n### Response {i}:\n{formatted_resp}\n"

        judge_prompt = f"""You are an expert evaluator. Your task is to evaluate multiple responses to a message and assign each a quality score from 0-100.

Consider the following criteria:
- Accuracy and correctness
- Helpfulness and completeness
- Clarity and coherence
- Relevance to the message and conversation context


## Conversation History:
{history_context}

## Current Message:
{input_text}

## Responses to Evaluate:
{responses_text}

## Instructions:
Provide a score (0-100) for each response. Output ONLY the scores in this exact format:
Response 1: [score]
Response 2: [score]
Response 3: [score]
... etc

Do not include any other text or explanation."""

        # Log the judge prompt
        print("\n" + "=" * 80)
        print("2. JUDGE PROMPT")
        print("=" * 80)
        print(judge_prompt)

        # Query judge LLM
        judge_result = await self.get_chat_result_fn(
            runnable=self.judge_llm,
            stream=False,
            input_value=judge_prompt,
            system_message=self.judge_system_message,
        )

        # Log the judge response
        print("\n" + "=" * 80)
        print("3. JUDGE RESPONSE")
        print("=" * 80)
        print(judge_result.text)

        # Parse the scores
        scores = self._parse_judge_scores(judge_result.text, len(responses))

        return scores

    def _parse_judge_scores(self, judge_text: str, expected_count: int) -> list[float]:
        """Parse scores from judge LLM response.

        Args:
            judge_text: Raw text from judge LLM
            expected_count: Expected number of scores

        Returns:
            List of scores (defaults to 0.0 if parsing fails)
        """
        import re

        scores = []

        # Try to extract scores in format "Response N: score"
        pattern = r"Response\s+\d+:\s*(\d+\.?\d*)"
        matches = re.findall(pattern, judge_text, re.IGNORECASE)

        if matches and len(matches) == expected_count:
            for score_str in matches:
                try:
                    score = float(score_str)
                    score = max(0.0, min(100.0, score))  # Normalize to 0-100
                    scores.append(score)
                except ValueError:
                    scores.append(0.0)
        else:
            # Fallback: try to extract all numbers
            all_numbers = re.findall(r"\d+\.?\d*", judge_text)
            if len(all_numbers) >= expected_count:
                for i in range(expected_count):
                    try:
                        score = float(all_numbers[i])
                        score = max(0.0, min(100.0, score))
                        scores.append(score)
                    except (ValueError, IndexError):
                        scores.append(0.0)
            else:
                # If parsing completely fails, assign equal scores
                print("⚠️ Failed to parse judge scores, assigning default scores")
                scores = [50.0] * expected_count

        return scores
