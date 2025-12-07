from typing import Any, Optional
import pandas as pd
import random

from code.ensemble.ensemble_base import EnsembleBase
from code.models.llm_judge import LLMJudge
from code.technical.content import TextContent
from code.technical.response_schema import GeneralEnsembleSchema
from code.technical.utils import get_field
    

class ReasoningEnsemble(EnsembleBase):
    def __init__(self, dataset_name, members_configuration, run_missing = True, type_name = "reasoning", judge_model: Optional[Any] = None):
        super().__init__(dataset_name, members_configuration, run_missing, type_name)
        self.llm = judge_model if judge_model is not None else LLMJudge()

    def evaluate_single_problem(self, problem_id):
        single_problem_df = self.answers[self.answers["problem_id"] == problem_id].copy()

        if single_problem_df.empty:
            self.logger.warning(f"No answers for problem {problem_id}")
            return None

        answer_list = single_problem_df["answer"].tolist()
        reasoning_list = single_problem_df["reasoning"].tolist()

        final_answer = self.evaluate_reasoning_using_llm(answer_list, reasoning_list)
        return final_answer
        
    def evaluate_reasoning_using_llm(self, answer_list, reasoning_list):
        problem_description = self.config.get("problem_description_prompt", "")
        reasoning_prompt_path = "prompts/ensemble/ensemble_reasoning_main.txt"
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
            [TextContent(prompt_filled)],
            response_schema=schema,
        )

        final_answer = get_field(response, "final_answer")
    
        return final_answer