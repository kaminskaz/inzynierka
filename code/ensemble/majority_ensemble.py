import pandas as pd
import random

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
        
        else:
            if "answer" not in single_problem_df.columns:
                self.logger.error(f"'answer' column missing for problem {problem_id}")
                return None
            
            counts = single_problem_df["answer"].value_counts()

            max_count = counts.max()
            tied_answers = counts[counts == max_count].index.tolist()
            most_popular_answer = random.choice(tied_answers)
            return most_popular_answer
            

    def evaluate(self):
        results = []
        problem_ids = self.answers["problem_id"].unique()

        for problem_id in problem_ids:
            final_answer = self.evaluate_single_problem(problem_id)
            results.append({
                "problem_id": problem_id,
                "ensemble_answer": final_answer
            })

        results_df = pd.DataFrame(results)
        return results_df
        

    def evaluate_majority_using_llm(self):
        pass