from code.technical.content import ImageContent, TextContent
from traitlets import List
from code.strategies.strategybase import StrategyBase
from code.technical.response_schema import DescriptionResponseSchema, ResponseSchema

class DescriptiveStrategy(StrategyBase):
    def run(self):
        descriptions = []
        for i in range(self.config.num_questions):
            choice_image_input = self.get_choice_image(index=i)
            descriptions_prompt = self.get_description_prompt()
            contents_to_send_descriptions = [
                TextContent(descriptions_prompt),

                ImageContent(choice_image_input)
            ]
            description_response = self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema) 
            descriptions.append(description_response)
        self.save_descriptions_to_textfile(descriptions)
        separator = "\n"  
        all_descriptions_text = separator.join(descriptions)

        prompt = self.get_prompt(self.dataset_name, self.strategy_name)
        if self.config.category == 'BP' or self.config.category == 'choice_only':
            contents_to_send = [
                    TextContent(prompt + "\n\nDescriptions:\n" + all_descriptions_text)
            ]
        else:
            image_input = self.get_choice_panel()
            contents_to_send = [
                    TextContent(prompt + "\n\nDescriptions:\n" + all_descriptions_text),
                    ImageContent(image_input)
            ]
        response = self.model.ask_structured(contents_to_send, schema=ResponseSchema)
        self.save_raw_answers_to_csv(response)
        self.save_metadata()
        self.logger.info(f"Descriptive strategy run completed for dataset: {self.dataset_name}, strategy: {self.strategy_name} using model: {self.model.name}")

    def get_description_prompt(self) -> str:
        # Implement prompt generation for descriptions here
        return "Please describe the following image in detail."

    def save_descriptions_to_textfile(self, descriptions: List[str]):
        with open(f"data/{self.dataset_name}/descriptions.txt", "w") as f:
            for desc in descriptions:
                f.write(f"{desc}\n")