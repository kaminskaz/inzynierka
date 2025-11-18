from code.ensemble.ensemble_base import EnsembleBase

class MajorityEnsemble(EnsembleBase):
    def evaluate_single_problem(self, problem_id):
        single_problem_df = self.answers[self.answers["problem_id"] == problem_id].copy()

        if single_problem_df.empty:
            self.logger.warning(f"No answers for problem {problem_id}")
            return None

        if self.dataset_config.category == "BP":
            single_problem_df["answer"] = (
                single_problem_df["left_side_rule"].astype(str)
                + " vs. "
                + single_problem_df["right_side_rule"].astype(str)
            )

            answer_list = single_problem_df["answer"].tolist()

            final_answer = self.evaluate_majority_using_llm(answer_list)
            return final_answer

    
    def evaluate(self):
        pass

    def evaluate_majority_using_llm(self):
        pass