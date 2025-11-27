import pandas as pd
import random

from code.ensemble.ensemble_base import EnsembleBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import GeneralEnsembleSchema

class ReasoningEnsemble(EnsembleBase):
    def evaluate_single_problem(self, problem_id):
        single_problem_df = self.answers[self.answers["problem_id"] == problem_id].copy()

        if single_problem_df.empty:
            self.logger.warning(f"No answers for problem {problem_id}")
            return None

        answer_list = single_problem_df["answer"].tolist()
        reasoning_list = single_problem_df["reasoning"].tolist()
        image_path = (f"data/{self.dataset}/problems/{problem_id}/question_panel.png")

        final_answer = self.evaluate_reasoning_using_llm(answer_list, reasoning_list, question_image_path=image_path)
        return final_answer
        
    def evaluate_reasoning_using_llm(self, answer_list, reasoning_list, question_image_path):
        problem_description = self.config.get("problem_description_prompt", "")
        reasoning_prompt_path = "prompts/ensemble/ensemble_reasoning_with_image_main.txt"
        sample_answer =self.config.get("sample_answer_structure", "")
        with open(reasoning_prompt_path, "r", encoding="utf-8") as file:
            reasoning_prompt = file.read()
   
        all_answers_str = "\n".join(f"- {ans} (reasoning: {reas})" for ans, reas in zip(answer_list, reasoning_list))

        schema = GeneralEnsembleSchema
        prompt_filled = reasoning_prompt.format(
            problem_description=problem_description,
            all_answers=all_answers_str,
            sample_answer=sample_answer
        )

        response = self.llm.ask(
            [TextContent(prompt_filled), ImageContent(question_image_path)],
            response_schema=schema,
        )

        return response.final_answer