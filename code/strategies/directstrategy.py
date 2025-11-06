from code.strategies.strategybase import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import ResponseSchema

class DirectStrategy(StrategyBase):
    def run(self):
        image_input = self.get_question_panel()
        prompt = self.get_prompt(self.dataset_name, self.strategy_name)
        contents_to_send = [
            TextContent(prompt),
            ImageContent(image_input)
        ]
        response = self.model.ask_structured(contents=contents_to_send, schema=ResponseSchema)
        self.save_raw_answers_to_csv(response)
        self.save_metadata()
        self.logger.info(f"Direct strategy run completed for dataset: {self.dataset_name}, strategy: {self.strategy_name} using model: {self.model.name}")  