import pandas as pd
from datasets import Dataset

class MultiLabelMultiClassDataset:

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

        self.train_data = self.load_data(self.args.train_data_path)
        self.val_data = self.load_data(self.args.val_data_path)
        self.test_data = self.load_data(self.args.test_data_path)

        self.train_dataset = self.generate_dataset(self.train_data)
        self.val_dataset = self.generate_dataset(self.val_data)
        self.test_dataset = self.generate_dataset(self.test_data)

        self.label_mappings = {
            label_name: {label: i for i, label in enumerate(self.train_data[label_name].unique())} 
            for label_name in self.args.label_names
        }
        self.id_mappings = {
            label_name: {i: label for i, label in self.label_mappings[label_name].items()} 
            for label_name in self.args.label_names
        }

    def load_data(self, path:str) -> pd.DataFrame:
        if path is None:
            return None
        data = pd.read_csv(path)
        data = self.preprocess(data)
        return data
    
    def preprocess(self, data:pd.DataFrame) -> pd.DataFrame:
        if data is None:
            return None
        # Split aspect column into multiple columns
        data[self.aspect_name] = data[self.aspect_name].apply(self.split_aspect)
        for label_name in self.args.label_names:
            data[label_name] = data[label_name].apply(lambda x: x[label_name])
        return data
    
    def split_aspect(self, aspect:str) -> dict:
        aspects = aspect.split("|")
        aspects = {aspect.split(":")[0]: float(aspect.split(":")[1]) for aspect in aspects}
        return aspects

    def generate_dataset(self, data:pd.DataFrame) -> Dataset:
        if data is None:
            return None
        dataset = Dataset.from_pandas(data)
        return dataset.map(self.tokenize_function, batched=True)

    def tokenize_function(self, examples:dict) -> dict:
        inputs = self.tokenizer(
            examples[self.args.text_name], truncation=True, max_length=self.args.max_length, padding="max_length"
        )
        for label_name in self.args.label_names:
            inputs[label_name] = [self.label_to_id(label) for label in examples[label_name]]
        return inputs

    def num_labels(self, label_name:str) -> int:
        return len(self.label_mappings[label_name])

    def label_to_id(self, label_name:str, label:str) -> int:
        return self.label_mappings[label_name][label]

    def id_to_label(self, label_name:str, id:int) -> str:
        return self.id_mappings[label_name][id]