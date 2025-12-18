
import os
import sys

from src.models.llm_judge import LLMJudge
from src.technical.content import ImageContent, TextContent
from src.technical.response_schema import BongardEvaluationSchema


def main():
    print("Preparing LLM", flush=True)
    llm = LLMJudge()

    answer1 = TextContent("The capital of Norway is Bergen.")
    key1 = TextContent("The capital of Norway is Oslo.")
    prompt = (
        "Evaluate the similarity between the provided answer and the key answer."
        "Respond with a similarity label and provide reasoning for your judgment."
    )
    response1 = llm.evaluate_similarity(
        prompt, answer1.text, key1.text, response_schema=BongardEvaluationSchema
    )
    print("Response (text):", response1, flush=True)

    answer2 = TextContent("Oslo is the capital of Norway.")
    key2 = TextContent("The capital of Norway is Oslo.")
    response2 = llm.evaluate_similarity(
        prompt, answer2.text, key2.text, response_schema=BongardEvaluationSchema
    )
    print("Response (text):", response2, flush=True)

    llm.stop()


if __name__ == "__main__":
    main()
