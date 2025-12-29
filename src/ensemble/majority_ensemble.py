import pandas as pd
import random

from typing import Optional, Any, List
from src.ensemble.ensemble_base import EnsembleBase
from src.technical.content import TextContent
from src.technical.response_schema import GeneralEnsembleSchema
from src.technical.utils import get_field, get_dataset_config
from src.models.llm_judge import LLMJudge
from string import Template

class MajorityEnsemble(EnsembleBase):
    def __init__(
            self, 
            dataset_name, 
            members_configuration, 
            skip_missing = True, 
            type_name = "majority", 
            judge_model: Optional[LLMJudge] = None,
            prompt_number: Optional[int] = 1
            ):
        super().__init__(dataset_name, members_configuration, skip_missing, type_name, prompt_number)
        if get_dataset_config(dataset_name).category == "BP":
            self.llm = judge_model if judge_model is not None else LLMJudge()
            self.config["ensemble_model"] = self.llm.get_model_name()

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
        first_member = next(
            v for k, v in self.config.items() if k.startswith("member_")
        )

        sample_answer = first_member.get("sample_answer_prompt", "")
        problem_description = first_member.get("problem_description_prompt", "")
        majority_prompt_path = self.get_ensemble_prompt_path(prompt_number=self.prompt_number)
        with open(majority_prompt_path, "r", encoding="utf-8") as file:
            majority_prompt = file.read()
        
        all_answers_str = "\n".join(f"- {ans}" for ans in answer_list)

        template = Template(majority_prompt)

        schema = GeneralEnsembleSchema
        prompt_filled = template.substitute(
            problem_description=problem_description,
            all_answers=all_answers_str,
            sample_answer=sample_answer
        )
        response = self.llm.ask(
            [TextContent(prompt_filled)],
            schema=schema,
        )

        final_answer = get_field(response, "final_answer")
        return final_answer


        