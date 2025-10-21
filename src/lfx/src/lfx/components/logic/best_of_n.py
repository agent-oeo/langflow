"""Best-of-N inference-time scaling algorithm implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lfx.schema.message import Message

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


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

    def __init__(self, judge_llm: BaseChatModel, get_chat_result_fn: Any, logger_fn: Any = None) -> None:
        """Initialize Best-of-N algorithm.

        Args:
            judge_llm: Language model to use for judging/scoring responses
            get_chat_result_fn: Async function to get chat results from a model
            logger_fn: Optional function to log messages to component logs
        """
        self.judge_llm = judge_llm
        self.get_chat_result_fn = get_chat_result_fn
        self.logger_fn = logger_fn

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

        # Log all generated responses
        if self.logger_fn:
            self.logger_fn("\n" + "=" * 80)
            self.logger_fn("1. GENERATED RESPONSES")
            self.logger_fn("=" * 80)
            for i, response in enumerate(responses, 1):
                self.logger_fn(f"\n--- Response {i} ---")
                self.logger_fn(response.text)

        # Score responses using judge LLM (all at once in a single prompt)
        scores = await self._score_responses(responses, input_value, conversation_history)

        # Select top N responses by score
        scored_responses = list(enumerate(scores))
        scored_responses.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored_responses[:top_n]]

        # Primary selection is the best one
        selected_index = top_indices[0]

        # Log final selection
        if self.logger_fn:
            self.logger_fn("\n" + "=" * 80)
            self.logger_fn("FINAL SELECTION")
            self.logger_fn("=" * 80)
            for i, score in enumerate(scores, 1):
                marker = " 🏆 SELECTED" if i - 1 == selected_index else ""
                self.logger_fn(f"Response {i}: Score {score:.1f}/100{marker}")

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
        else:
            input_text = str(original_input)

        # Build conversation history context
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            history_context = "\n## Conversation History:\n"
            for msg in conversation_history:
                sender = getattr(msg, "sender", "Unknown")
                history_context += f"**{sender}**: {msg.text}\n"
            history_context += "\n"

        # Build the judge prompt with all responses
        responses_text = ""
        for i, response in enumerate(responses, 1):
            responses_text += f"\n### Response {i}:\n{response.text}\n"

        judge_prompt = f"""You are an expert evaluator. Your task is to evaluate multiple responses to a question and assign each a quality score from 0-100.

Consider the following criteria:
- Accuracy and correctness
- Helpfulness and completeness
- Clarity and coherence
- Relevance to the question and conversation context

{history_context}## Current Question:
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
        if self.logger_fn:
            self.logger_fn("\n" + "=" * 80)
            self.logger_fn("2. JUDGE PROMPT")
            self.logger_fn("=" * 80)
            self.logger_fn(judge_prompt)

        # Query judge LLM
        judge_result = await self.get_chat_result_fn(
            runnable=self.judge_llm,
            stream=False,
            input_value=judge_prompt,
            system_message="You are an objective response evaluator. Provide only numeric scores in the requested format.",
        )

        # Log the judge response
        if self.logger_fn:
            self.logger_fn("\n" + "=" * 80)
            self.logger_fn("3. JUDGE RESPONSE")
            self.logger_fn("=" * 80)
            self.logger_fn(judge_result.text)

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
                if self.logger_fn:
                    self.logger_fn("⚠️ Failed to parse judge scores, assigning default scores")
                scores = [50.0] * expected_count

        return scores
