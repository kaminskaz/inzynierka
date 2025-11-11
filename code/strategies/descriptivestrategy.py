from traitlets import List
import os
import json

from code.technical.content import ImageContent, TextContent
from code.strategies.strategybase import StrategyBase
from code.technical.response_schema import DescriptionResponseSchema, ResponseSchema

class DescriptiveStrategy(StrategyBase):
    def run_single_problem(self, problem_id: str, descriptions_prompt: str, main_prompt: str) -> ResponseSchema:
        descriptions = []
        problem_descriptions_dict = {}

        for i in range(self.config.num_questions):
            letter=chr(65 + i)

            if self.config.category != 'BP':
                choice_image_input = self.get_choice_image(problem_id, index=letter)
                index_key = letter
            else:
                choice_image_input = self.get_choice_image(problem_id, index=i)
                index_key = i

            contents_to_send_descriptions = [
                TextContent(descriptions_prompt),
                ImageContent(choice_image_input)
            ]
            description_response = self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema) 
            raw_description = description_response.description
            if raw_description is not None and description_response.description is not None:
                description_response.description = f"{index_key}: {raw_description}"
                problem_descriptions_dict[index_key]=raw_description

            descriptions.append(description_response)

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
    
        return response, problem_descriptions_dict
    
    def run(self):
        output_csv = os.path.join("results", self.strategy_name, self.dataset_name, "results.csv")
        dataset_dir = os.path.join("data", self.dataset_name, "problems")
        descriptions_path = os.path.join("results", self.strategy_name, self.dataset_name, "descriptions.json")

        descriptions_prompt = self.get_prompt("describe_main")
        problem_description = self.get_prompt("problem_description_main")
        question_prompt = self.get_prompt("question_main")
        main_prompt = f'{problem_description}\n{question_prompt}'

        results = []
        all_descriptions_data={}

        for problem_entry in os.scandir(dataset_dir):
            try:
                if not problem_entry.is_dir():
                    continue
                problem_id = problem_entry.name
                image_path = self.get_choice_panel(problem_id)
                response, problem_descriptions = self.run_single_problem(problem_id, descriptions_prompt, main_prompt)

                if problem_descriptions:
                    all_descriptions_data[problem_id]=problem_descriptions

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
        self.save_descriptions_to_json(descriptions_path, all_descriptions_data)
        self.logger.info(f"Descriptive strategy run completed for dataset: {self.dataset_name}, strategy: {self.strategy_name} using model: {self.model.name}")

    def save_descriptions_to_json(self, descriptions_path: str, all_descriptions_data: dict):
        """
        Saves the collected descriptions dictionary to a JSON file.

        Args:
            descriptions_path (str): The full file path to save the JSON to.
            all_descriptions_data (dict): The dictionary containing problem IDs
                                          and their corresponding descriptions.
        """
        try:
            # Ensure the directory exists
            directory = os.path.dirname(descriptions_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            # Write the data to the JSON file
            with open(descriptions_path, 'w', encoding='utf-8') as f:
                json.dump(all_descriptions_data, f, indent=4)
                
        except (IOError, OSError) as e:
            self.logger.error(f"Failed to create directory or write to file {descriptions_path}: {e}")
        except TypeError as e:
            self.logger.error(f"Error serializing descriptions to JSON: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in save_descriptions_to_json: {e}")