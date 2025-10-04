from dotenv import load_dotenv
import os
from huggingface_hub import login, snapshot_download
from code.preprocessing.bongard.BPprocessor import BPProcessor
from code.preprocessing.vcog.CVRprocessor import CVRProcessor
from code.preprocessing.vcog.MARSprocessor import MARSProcessor
from code.preprocessing.vcog.RAVENprocessor import RAVENProcessor

load_dotenv()

class DataProcessor:
    def __init__(self, load: bool = False):
        self.load = load
        self.raw_data_folder_path = "data_raw"
        self.bp_processor = BPProcessor()
        self.cvr_processor = CVRProcessor()
        self.mars_processor = MARSProcessor()
        self.raven_processor = RAVENProcessor()

    def process(self):
        if self.load:
            self.download_data_from_huggingface("vcog/vcog-bench", repo_type="dataset")
            print("Data loaded.")
        self.bp_processor.process()
        print("Bongard problems processed.")
        self.cvr_processor.process()
        print("CVR problems processed.")
        self.mars_processor.process()
        print("MARS problems processed.")
        self.raven_processor.process()
        print("RAVEN problems processed.")

    def download_data_from_huggingface(self, repo_id: str, repo_type: str = "dataset"):

        login(token=os.getenv("HF_API_TOKEN"))

        if not os.path.exists(self.raw_data_folder_path):
            os.makedirs(self.raw_data_folder_path)

        data_path = os.path.join(self.raw_data_folder_path, repo_id.split("/")[-1])

        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=data_path
        )

        print(f"Dataset downloaded to {data_path}")