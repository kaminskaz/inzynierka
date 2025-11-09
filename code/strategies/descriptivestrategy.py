from traitlets import List
import os

from code.technical.content import ImageContent, TextContent
from code.strategies.strategybase import StrategyBase
from code.technical.response_schema import DescriptionResponseSchema, ResponseSchema

class DescriptiveStrategy(StrategyBase):
    def run_single_problem(self, problem_id: str, descriptions_prompt: str, main_prompt: str) -> ResponseSchema:
        descriptions = []
        for i in range(self.config.num_questions):
            choice_image_input = self.get_choice_image(problem_id, index=i)
            contents_to_send_descriptions = [
                TextContent(descriptions_prompt),
                ImageContent(choice_image_input)
            ]
            description_response = self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema) 
            descriptions.append(description_response)

        # self.save_descriptions_to_textfile(descriptions) 
        all_descriptions_text = "\n\n".join([r.description for r in descriptions if r is not None])

        prompt = f'{main_prompt}\nDescriptions:\n{all_descriptions_text}'

        if self.config.category == 'BP' or self.config.category == 'choice_only':
            contents_to_send = [
                    TextContent(prompt)
            ]
        else:
            image_input = self.get_choice_panel(problem_id)
            contents_to_send = [
                    TextContent(prompt),
                    ImageContent(image_input)
            ]
            
        response = self.model.ask_structured(contents_to_send, schema=ResponseSchema)
    
        return response
    
    def run(self):
        output_csv = os.path.join("results", self.strategy_name, self.dataset_name, "results.csv")
        dataset_dir = os.path.join("data", self.dataset_name, "problems")

        descriptions_prompt = self.get_prompt("describe_main", self.strategy_name, self.dataset_name)
        problem_description = self.get_prompt("problem_description_main", self.strategy_name, self.dataset_name)
        question_prompt = self.get_prompt("question_main", self.strategy_name, self.dataset_name)
        main_prompt = f'{problem_description}\n{question_prompt}'

        results = []

        for problem_entry in os.scandir(dataset_dir):
            try:
                if not problem_entry.is_dir():
                    continue
                problem_id = problem_entry.name
                image_path = self.get_choice_panel(problem_id)
                response = self.run_single_problem(problem_id, descriptions_prompt, main_prompt)

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
        self.logger.info(f"Descriptive strategy run completed for dataset: {self.dataset_name}, strategy: {self.strategy_name} using model: {self.model.name}")

    # TODO: adjust path and saving mechanism
    def save_descriptions_to_textfile(self, descriptions: List[str]):
        with open(f"data/{self.dataset_name}/descriptions.txt", "w") as f:
            for desc in descriptions:
                if desc and hasattr(desc, "description"):
                    f.write(desc.description.strip() + "\n\n")