from typing import Any, Optional
import pandas as pd
import random

from src.ensemble.ensemble_base import EnsembleBase
from src.models.llm_judge import LLMJudge
from src.technical.content import TextContent
from src.technical.response_schema import GeneralEnsembleSchema
from src.technical.utils import get_field
from string import Template
    

class ReasoningEnsemble(EnsembleBase):
    def __init__(
            self, 
            dataset_name, 
            members_configuration, 
            skip_missing = True, 
            type_name = "reasoning", 
            judge_model: Optional[LLMJudge] = None,
            prompt_number: Optional[int] = 1
            ):
        super().__init__(dataset_name, members_configuration, skip_missing, type_name, prompt_number)
        self.llm = judge_model if judge_model is not None else LLMJudge()
        self.config["ensemble_model"] = self.llm.get_model_name()

    def evaluate_single_problem(self, problem_id):
        single_problem_df = self.answers[self.answers["problem_id"] == problem_id].copy()

        if single_problem_df.empty:
            self.logger.warning(f"No answers for problem {problem_id}")
            return None

        answer_list = single_problem_df["answer"].tolist()
        reasoning_list = single_problem_df["rationale"].tolist()

        final_answer, rationale = self.evaluate_reasoning_using_llm(answer_list, reasoning_list)
        return final_answer, rationale
        
    def evaluate_reasoning_using_llm(self, answer_list, reasoning_list):
        all_answers_str = "\n".join(f"- {ans} (reasoning: {reas})" for ans, reas in zip(answer_list, reasoning_list))
        prompt_filled = self._get_filled_prompt(all_answers=all_answers_str)
        
        schema = GeneralEnsembleSchema
        response = self.llm.ask(
            [TextContent(prompt_filled)],
            schema=schema,
        )

        final_answer = get_field(response, "final_answer")
        rationale = get_field(response, "rationale")
    
        return final_answer, rationale