import os

from code.strategies.strategybase import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import ResponseSchema

class ClassificationStrategy(StrategyBase):
    def run_single_problem(self, image_path: str, prompt: str) -> ResponseSchema:

        contents_to_send = [
            TextContent(prompt),
            ImageContent(image_path)
        ]

        response = self.model.ask_structured(contents=contents_to_send, schema=ResponseSchema)
        self.save_raw_answers_to_csv(response)
        self.save_metadata()
        self.logger.info(f"Direct strategy run completed for dataset: {self.dataset_name}, strategy: {self.strategy_name} using model: {self.model.name}") 

    def run(self):
        problem_descr = self.get_prompt("problem_description_main")
        question_prompt = self.get_prompt("question_main")
        prompt = f"{problem_descr}\n{question_prompt}"

        output_csv = os.path.join("results", self.strategy_name, self.dataset_name, "results.csv")
        dataset_dir = os.path.join("data", self.dataset_name, "problems")

        results = []

        for problem_entry in os.scandir(dataset_dir):
            try:
                if not problem_entry.is_dir():
                    continue
                problem_id = problem_entry.name
                image_path = self.get_classification_panel(problem_id)
                response = self.run_single_problem(image_path, prompt)

                if response:
                    result = {
                        "image": image_path.name,
                        "answer": response.answer,
                        "confidence": response.confidence,
                        "rationale": response.rationale
                    }
                else:
                    result = {
                    "image": image_path.name,
                    "answer": "",
                    "confidence": "",
                    "rationale": ""
                    }

                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing {image_path.name}: {e}")

        self.save_raw_answers_to_csv(results, output_csv)
        self.save_metadata()
        self.logger.info(f"Classification strategy run completed for dataset: {self.dataset_name}, strategy: {self.strategy_name} using model: {self.model.name}")   