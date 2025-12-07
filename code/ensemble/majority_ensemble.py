import pandas as pd
import random

from typing import Optional, Any, List
from code.ensemble.ensemble_base import EnsembleBase
from code.technical.content import TextContent
from code.technical.response_schema import GeneralEnsembleSchema
from code.technical.utils import get_field
from code.models.llm_judge import LLMJudge

class MajorityEnsemble(EnsembleBase):
    def __init__(self, dataset_name, members_configuration, run_missing = True, type_name = "majority", judge_model: Optional[Any] = None):
        super().__init__(dataset_name, members_configuration, run_missing, type_name)
        self.llm = judge_model if judge_model is not None else LLMJudge()

    def evaluate_single_problem(self, problem_id):
        single_problem_df = self.answers[self.answers["problem_id"] == problem_id].copy()

        if single_problem_df.empty:
            self.logger.warning(f"No answers for problem {problem_id}")
            return None

        if self.dataset_config.category == "BP":
            answer_list = single_problem_df["answer"].tolist()

            final_answer = self.evaluate_majority_using_llm(answer_list)
            return final_answer
        
        else:
            if "answer" not in single_problem_df.columns:
                self.logger.error(f"'answer' column missing for problem {problem_id}")
                return None
            
            counts = single_problem_df["answer"].value_counts()

            max_count = counts.max()
            tied_answers = counts[counts == max_count].index.tolist()
            most_popular_answer = random.choice(tied_answers)
            return most_popular_answer
        
    def evaluate_majority_using_llm(self, answer_list):
        problem_description = self.config.get("problem_description_prompt", "")
        sample_answer =self.config.get("sample_answer_structure", "")
        majority_prompt_path = "prompts/ensemble/ensemble_majority_main.txt"
        with open(majority_prompt_path, "r", encoding="utf-8") as file:
            majority_prompt = file.read()
        
        all_answers_str = "\n".join(f"- {ans}" for ans in answer_list)

        schema = GeneralEnsembleSchema
        prompt_filled = majority_prompt.format(
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


        