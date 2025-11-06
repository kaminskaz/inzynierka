from traitlets import List
from code.strategies.strategybase import StrategyBase
class DescriptiveStrategy(StrategyBase):
    def run(self):
        descriptions = []
        # Implement the descriptive strategy here
        # describe all choice images in a loop and save those results to a single text file with proper formatting 
        for i in range(self.config.num_questions):
            choice_image_input = self.get_choice_image(index=i)
            descriptions_prompt = self.get_description_prompt()
            description_response = self.model.ask(descriptions_prompt, image=choice_image_input) #TODO: Karolina - adapt to model interface
            descriptions.append(description_response)
        self.save_descriptions_to_textfile(descriptions)

        image_input = self.get_question_panel()
        prompt = self.get_prompt(self.dataset_name, self.strategy_name)
        response = self.model.ask(prompt, image=image_input) #TODO: Karolina - adapt to model interface

    def get_description_prompt(self) -> str:
        # Implement prompt generation for descriptions here
        pass

    def save_descriptions_to_textfile(self, descriptions: List[str]):
        with open(f"data/{self.dataset_name}/descriptions.txt", "w") as f:
            for desc in descriptions:
                f.write(f"{desc}\n")