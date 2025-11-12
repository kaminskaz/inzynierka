import asyncio
import os
import sys
from pydantic import create_model

from code.models.llm_judge import LLMJudge
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import SimilarityResponseSchema


async def main():
    print("Preparing LLM", flush=True)
    llm = LLMJudge()

    answer1 = TextContent("The capital of Norway is Bergen.")
    key1 = TextContent("The capital of Norway is Oslo.")
    response1 = await llm.evaluate_similarity(
        answer1.text, key1.text, response_schema=SimilarityResponseSchema
    )
    print("Response (text):", response1, flush=True)

    answer2 = TextContent("Oslo is the capital of Norway.")
    key2 = TextContent("The capital of Norway is Oslo.")
    response2 = await llm.evaluate_similarity(
        answer2.text, key2.text, response_schema=SimilarityResponseSchema
    )
    print("Response (text):", response2, flush=True)

    llm.stop()


if __name__ == "__main__":
    asyncio.run(main())
