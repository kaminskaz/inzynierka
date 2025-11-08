from traitlets import List
from code.strategies.strategybase import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import DescriptionResponseSchema, ResponseSchema

class ContrastiveStrategy(StrategyBase):
    def run(self):
        if self.config.category == 'BP' or self.config.category == 'choice_only': 
            descriptions = []
            
            for i in range(self.config.num_questions):
                if self.config.category == 'BP':
                    choice_image_input = self.get_choice_image(index=i)
                else:
                    choice_image_input = self.get_blackout_image(index=i)
                
                descriptions_prompt = self.get_prompt("describe_main", self.strategy_name, self.dataset_name)
                contents_to_send_descriptions = [
                    TextContent(descriptions_prompt),
                    ImageContent(choice_image_input)
                ]
                description_response = self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema) 
                descriptions.append(description_response)

            self.save_descriptions_to_textfile(descriptions)
        
            all_descriptions_text = "\n\n".join([r.description for r in descriptions if r is not None])

            problem_description = self.get_prompt("problem_description_main", self.strategy_name, self.dataset_name)
            question_prompt = self.get_prompt("question_main", self.strategy_name, self.dataset_name)

            prompt = f'{problem_description}\n{question_prompt}\nDescriptions:\n{all_descriptions_text}'
            
            contents_to_send = [
                    TextContent(prompt)
            ]

        else:
            question_panel_input = self.get_question_panel()
            descriptions_prompt = self.get_prompt("describe_main", self.strategy_name, self.dataset_name)

            contents_to_send_descriptions = [
                    TextContent(descriptions_prompt),
                    ImageContent(question_panel_input)
            ]

            description_response = self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema) 
            self.save_descriptions_to_textfile([description_response])

            problem_description = self.get_prompt("problem_description_main", self.strategy_name, self.dataset_name)
            question_prompt = self.get_prompt("question_main", self.strategy_name, self.dataset_name)

            prompt = f'{problem_description}\n{question_prompt}\nDescriptions:\n{all_descriptions_text}'
            choice_panel_input = self.get_choice_panel()

            contents_to_send = [
                    TextContent(prompt),
                    ImageContent(choice_panel_input)
            ]
        
        response = self.model.ask_structured(contents_to_send, schema=ResponseSchema)
        self.save_raw_answers_to_csv(response)
        self.save_metadata()
        self.logger.info(f"Descriptive strategy run completed for dataset: {self.dataset_name}, strategy: {self.strategy_name} using model: {self.model.name}")


    def save_descriptions_to_textfile(self, descriptions: List[str]):
        with open(f"data/{self.dataset_name}/descriptions.txt", "w") as f:
            for desc in descriptions:
                if desc and hasattr(desc, "description"):
                    f.write(desc.description.strip() + "\n\n")