from traitlets import List
import os

from code.strategies.strategybase import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import DescriptionResponseSchema, ResponseSchema, BPDescriptionResponseSchemaContrastive

class ContrastiveStrategy(StrategyBase):

    def run_single_problem(self, problem_id: str, descriptions_prompt: str, main_prompt: str) -> ResponseSchema:
        
        problem_descriptions_dict = {}

        if self.config.category == 'BP' or self.config.category == 'choice_only': 
            descriptions = []
            
            for i in range(self.config.num_choices):
                if self.config.category == 'BP':
                    if i >= self.config.num_choices // 2:
                        break
                    choice_image_input_1 = self.get_choice_image(problem_id, index=i)
                    choice_image_input_2 = self.get_choice_image(problem_id, index=i+6)
                    contents_to_send_descriptions = [
                        TextContent(descriptions_prompt),
                        ImageContent(choice_image_input_1),
                        ImageContent(choice_image_input_2)
                    ]
                    description_response = self.model.ask_structured(contents_to_send_descriptions, schema=BPDescriptionResponseSchemaContrastive)

                    if description_response and description_response.description:
                        key = f"{i}"
                        problem_descriptions_dict[key] = description_response.description

                else: 
                    choice_image_input = self.get_blackout_image(problem_id, index=i)
                    contents_to_send_descriptions = [
                        TextContent(descriptions_prompt),
                        ImageContent(choice_image_input)
                    ]
                    description_response = self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema)
         
                    if description_response and description_response.description:
                        problem_descriptions_dict[i] = description_response.description
 
                descriptions.append(description_response)
        
            all_descriptions_text = "\n\n".join([r.description for r in descriptions if r is not None])

            prompt = f'{main_prompt}\nDescriptions:\n{all_descriptions_text}'
            
            contents_to_send = [
                    TextContent(prompt)
            ]

        else:
            question_panel_input = self.get_question_panel(problem_id)

            contents_to_send_descriptions = [
                    TextContent(descriptions_prompt),
                    ImageContent(question_panel_input)
            ]

            description_response = self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema) 
            
            if description_response and description_response.description:
                problem_descriptions_dict["question_panel"] = description_response.description
                all_descriptions_text = description_response.description
            else:
                all_descriptions_text = ""

            prompt = f'{main_prompt}\nDescriptions:\n{all_descriptions_text}'
            choice_panel_input = self.get_choice_panel(problem_id)

            contents_to_send = [
                    TextContent(prompt),
                    ImageContent(choice_panel_input)
            ]

        response = self.model.ask_structured(contents_to_send, schema=ResponseSchema)
        
        return response, problem_descriptions_dict
    

    def run(self):
        dataset_dir = os.path.join("data", self.dataset_name, "problems")
        descriptions_path = os.path.join(self.results_dir, "descriptions.json")
        all_descriptions_data = {}

        descriptions_prompt = self.get_prompt("describe_main")
        problem_description = self.get_prompt("problem_description_main")
        question_prompt = self.get_prompt("question_main")
        main_prompt = f'{problem_description}\n{question_prompt}'

        results = []

        for problem_entry in os.scandir(dataset_dir):
            try:
                if not problem_entry.is_dir():
                    continue
                problem_id = problem_entry.name
                image_path = self.get_choice_panel(problem_id)
            
                response, problem_descriptions = self.run_single_problem(problem_id, descriptions_prompt, main_prompt)

                if problem_descriptions:
                    all_descriptions_data[problem_id] = problem_descriptions

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

        self.save_raw_answers_to_csv(results)
        self.save_metadata(question_prompt=question_prompt, problem_description_prompt=problem_description, describe_prompt=descriptions_prompt)
        self.save_descriptions_to_json(descriptions_path, all_descriptions_data)
        self.logger.info(f"Descriptive strategy run completed for dataset: {self.dataset_name}, strategy: {self.strategy_name} using model: {self.model.get_model_name()}")