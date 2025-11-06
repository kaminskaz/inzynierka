from code.strategies.strategybase import StrategyBase

class DirectStrategy(StrategyBase):
    def run(self):
        image_input = self.get_question_panel()
        prompt = self.get_prompt(self.dataset_name, self.strategy_name)
        response = self.model.ask(prompt, image=image_input) #TODO: Karolina - adapt to model interface 
        self.save_raw_answers_to_csv(response)
        self.save_metadata()
        self.logger.info(f"Direct strategy run completed for dataset: {self.dataset_name}, strategy: {self.strategy_name} using model: {self.model.name}")  #TODO: Karolina - adapt to model interface 