import pandas as pd
import random
from typing import Dict, Any, List, Optional
from traitlets import List

from code.ensemble.ensemble_base import EnsembleBase
from code.models.vllm import VLLM
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import GeneralEnsembleSchema
from code.technical.utils import get_field
from string import Template


class ReasoningEnsembleWithImage(EnsembleBase):
    def __init__(self, dataset_name: str, members_configuration: List[List[str]], skip_missing: bool = True, judge_model: Optional[Any] = None, type_name: str = "reasoning_with_image"):
        super().__init__(dataset_name, members_configuration, skip_missing, type_name)
        self.vllm = judge_model if judge_model is not None else VLLM(model_name="OpenGVLab/InternVL3-8B")
        self.config["ensemble_model"] = self.vllm.get_model_name()

    def evaluate_single_problem(self, problem_id):
        single_problem_df = self.answers[self.answers["problem_id"] == problem_id].copy()

        if single_problem_df.empty:
            self.logger.warning(f"No answers for problem {problem_id}")
            return None

        answer_list = single_problem_df["answer"].tolist()
        reasoning_list = single_problem_df["rationale"].tolist()
        image_path = (f"data/{self.dataset_name}/problems/{problem_id}/question_panel.png")

        final_answer = self.evaluate_reasoning_using_llm(answer_list, reasoning_list, question_image_path=image_path)
        return final_answer
        
    def evaluate_reasoning_using_llm(self, answer_list, reasoning_list, question_image_path):
        problem_description = self.config.get("problem_description_prompt", "")
        reasoning_prompt_path = "prompts/ensemble/ensemble_reasoning_with_image_main.txt"
        sample_answer =self.config.get("sample_answer_structure", "")
        with open(reasoning_prompt_path, "r", encoding="utf-8") as file:
            reasoning_prompt = file.read()
   
        all_answers_str = "\n".join(f"- {ans} (reasoning: {reas})" for ans, reas in zip(answer_list, reasoning_list))
        template = Template(reasoning_prompt)

        schema = GeneralEnsembleSchema
        prompt_filled = template.substitute(
            problem_description=problem_description,
            all_answers=all_answers_str,
            sample_answer=sample_answer
        )

        response = self.vllm.ask(
            [TextContent(prompt_filled), ImageContent(question_image_path)],
            schema=schema,
        )

        final_answer = get_field(response, "final_answer")

        return final_answer