"""Best-of-N inference-time scaling algorithm implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lfx.schema.message import Message

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

DEFAULT_JUDGE_SYSTEM_MESSAGE = """
You are an expert evaluator. Compare all {num_responses} responses based on the given criteria.

Analyze each response and select the top {top_n} that best meet the criteria.

Provide your evaluation as a JSON object with this exact format:
{{
  "reasoning": "Your detailed reasoning explaining why you selected these specific responses, comparing their strengths and weaknesses against the criteria",
  "selected_indices": [list of exactly {top_n} indices]
}}

"""

DEFAULT_JUDGE_CRITERIA = """
Evaluation Criteria:
1. PROCESS AWARENESS: What stage are we at, and which candidate best advances the workflow towards completion of the user request?
   - Does the approach match the current stage (planning, data gathering, analysis, completion)?
   - Is the solution at current stage aligned with fulfiling user's request
2. STRATEGIC REASONING: 
   - Early stages: High-level sense of what direction is solution proceeding in, are the steps relevant
   - Data stages: Make sure the data manipulation or retrieval is done in a correct way based on information provided by the user
   - Complex requests: Step-by-step solution for complex requests in correct order
3. TOOL EXECUTION:
   - Is the tool appropriate for current stage of the process?
   - Is the tool useful for helping solve user's problem?
   - Are arguments correctly configured for stated goals or sub-tasks?
   - Do the steps build logically toward the objective?
Focus: Which candidate makes the best NEXT STEP toward successfully resolving the user's request?
"""



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
            else DEFAULT_JUDGE_SYSTEM_MESSAGE
        )
        self.top_n = 1
        self.budget = 4

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
        
        self.top_n = top_n
        self.budget = budget
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
        elif hasattr(original_input, "content"):
            # LangChain message - extract content
            content = original_input.content
            # If content is a list (multi-modal), extract text
            if isinstance(content, list):
                content = " ".join(
                    [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content]
                )
            input_text = str(content)

            # If AI message with tool calls, append tool call info
            if hasattr(original_input, "tool_calls") and original_input.tool_calls:
                import json

                tool_calls_text = "\n[Tool Calls]:\n"
                for tc in original_input.tool_calls:
                    tc_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
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
                if hasattr(msg, "content"):
                    # LangChain message - extract content
                    content = msg.content
                    # If content is a list (multi-modal), extract text
                    if isinstance(content, list):
                        content = " ".join(
                            [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content]
                        )
                    msg_text = str(content)

                    # Get role from message type
                    msg_type = type(msg).__name__
                    if "System" in msg_type:
                        sender = "System"
                    elif "Human" in msg_type or "User" in msg_type:
                        sender = "User"
                    elif "Tool" in msg_type:
                        sender = "Tool"
                        # For tool messages, also include tool name if available
                        if hasattr(msg, "name") and msg.name:
                            sender = f"Tool ({msg.name})"
                    elif "AI" in msg_type or "Assistant" in msg_type:
                        sender = "Assistant"
                        # For AI messages with tool calls, append tool call info
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            import json

                            tool_calls_text = "\n[Tool Calls]:\n"
                            for tc in msg.tool_calls:
                                tc_name = (
                                    tc.get("name", "unknown")
                                    if isinstance(tc, dict)
                                    else getattr(tc, "name", "unknown")
                                )
                                tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                tool_calls_text += f"  - {tc_name}({json.dumps(tc_args)})\n"
                            msg_text += tool_calls_text
                    else:
                        sender = "Unknown"
                elif hasattr(msg, "text"):
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
        for i, response in enumerate(responses):
            formatted_resp = format_message_with_tool_calls(response)
            responses_text += f"\n### Response {i}:\n{formatted_resp}\n"
        
        judge_context = """
        
## Conversation History:
{history_context}

## Current Message:
{input_text}

## Responses to Evaluate:
{responses_text}

"""

        judge_prompt = DEFAULT_JUDGE_CRITERIA + judge_context.format(history_context=history_context, 
                                                                     input_text=input_text,
                                                                     responses_text=responses_text)
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
            system_message=self.judge_system_message.format(num_responses=self.budget, top_n=self.top_n),
        )

        # Log the judge response
        print("\n" + "=" * 80)
        print("3. JUDGE RESPONSE")
        print("=" * 80)
        print(judge_result.text)

        # Parse the scores
        scores = self._parse_judge_scores(judge_result.text, self.budget)

        return scores

    def _parse_judge_scores(self, judge_text: str, expected_count: int) -> list[float]:
        """Parse scores from judge LLM response.

        Expects JSON format with selected_indices. Converts to scores where
        selected responses get 100 and others get 0.

        Args:
            judge_text: Raw text from judge LLM (JSON format)
            expected_count: Expected number of responses

        Returns:
            List of scores (100 for selected, 0 for others)
        """
        import json
        import re

        scores = [0.0] * expected_count

        try:
            # Try to parse as JSON
            # First, try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', judge_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{[^{}]*"selected_indices"[^{}]*\}', judge_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = judge_text

            result = json.loads(json_str)
            selected_indices = result.get("selected_indices", [])

            # Validate indices
            for idx in selected_indices:
                if isinstance(idx, int) and 0 <= idx < expected_count:
                    scores[idx] = 100.0
                else:
                    print(f"⚠️ Invalid index {idx} in selected_indices")

            # If we got valid selections, return
            if any(score > 0 for score in scores):
                return scores

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️ Failed to parse JSON from judge response: {e}")

        # Fallback: try old format "Response N: score"
        print("⚠️ Trying fallback parser for old score format")
        pattern = r"Response\s+\d+:\s*(\d+\.?\d*)"
        matches = re.findall(pattern, judge_text, re.IGNORECASE)

        if matches and len(matches) == expected_count:
            scores = []
            for score_str in matches:
                try:
                    score = float(score_str)
                    score = max(0.0, min(100.0, score))
                    scores.append(score)
                except ValueError:
                    scores.append(0.0)
            return scores

        # Final fallback: assign equal scores
        print("⚠️ All parsing failed, assigning equal scores to all responses")
        return [50.0] * expected_count
