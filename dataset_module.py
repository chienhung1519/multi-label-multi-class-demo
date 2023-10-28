import pandas as pd
from datasets import Dataset
import json
from pathlib import Path
from typing import List


class MultiLabelMultiClassDataset:

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

        # Load label mappings
        self.aspect2id = json.loads(Path(self.args.src_dir, "aspect_mapping.json").read_text())
        self.id2aspect = {v: k for k, v in self.aspect2id.items()}
        self.label2id = json.loads(Path(self.args.src_dir, "label_mapping.json").read_text())
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Load data
        self.train_data = self.load_data(self.args.train_data_path)
        self.eval_data = self.load_data(self.args.eval_data_path)
        self.test_data = self.load_data(self.args.test_data_path)

        # Generate datasets
        self.train_dataset = self.generate_dataset(self.train_data)
        self.eval_dataset = self.generate_dataset(self.eval_data)
        self.test_dataset = self.generate_dataset(self.test_data)
        self.train_dataset.shuffle(seed=self.args.seed)

    def load_data(self, path:str) -> pd.DataFrame:
        if path is None:
            return None
        data = pd.read_json(path)
        return self.preprocess(data)
    
    def preprocess(self, data:pd.DataFrame) -> pd.DataFrame:
        if data is None:
            return None
        # Split aspect column into multiple columns
        data["Aspects"] = data["Aspects"].apply(self.split_aspect)
        for aspect in list(self.aspect2id.keys()):
            data[aspect] = data["Aspects"].apply(
                lambda x: self.score_to_label(x[aspect]) if aspect in x else self.label2id["NoFillingIn"]
            )
        return self.split_aspects_to_examples(data)
    
    def split_aspect(self, raw_aspect:str) -> dict:
        if raw_aspect == "No filling in":
            return {}
        return {aspect[:-2]: int(aspect[-1:]) for aspect in raw_aspect.split("|")}

    def score_to_label(self, score:float) -> int:
        if score <= 3:
            return self.label2id["Negative"]
        else:
            return self.label2id["Positive"]
    
    def split_aspects_to_examples(self, data:pd.DataFrame) -> pd.DataFrame:
        if data is None:
            return None
        examples = []
        for _, row in data.iterrows():
            for aspect in self.aspect2id.keys():
                examples.append({
                    "ReviewTitle": row["ReviewTitle"],
                    "ReviewText": row["ReviewText"],
                    "aspect_ids": self.aspect_to_id(aspect),
                    "labels": row[aspect]
                })
        return pd.DataFrame(examples)

    def generate_dataset(self, data:pd.DataFrame) -> Dataset:
        if data is None:
            return None
        dataset = Dataset.from_pandas(data)
        dataset = dataset.map(self.tokenize_function, batched=True, remove_columns=["ReviewTitle", "ReviewText"])
        dataset.set_format(type="torch", columns=dataset.column_names)
        return dataset

    def tokenize_function(self, examples:dict) -> dict:
        inputs = self.tokenizer(
            f"{examples['ReviewTitle']}. {examples['ReviewText']}", truncation=True, max_length=self.args.max_seq_length, padding="max_length"
        )
        return inputs
    
    def train_dataset(self) -> Dataset:
        return self.train_dataset
    
    def eval_dataset(self) -> Dataset:
        return self.eval_dataset
    
    def test_dataset(self) -> Dataset:
        return self.test_dataset

    @property
    def num_labels(self) -> int:
        return len(self.label2id)
    
    @property
    def aspect_ids(self) -> List:
        return sorted(list(self.aspect2id.values()))
    
    @property
    def aspect_names(self) -> List:
        return sorted(list(self.aspect2id.keys()))

    def label_to_id(self, label:str) -> int:
        return self.label2id[label]

    def id_to_label(self, id:int) -> str:
        return self.id2label[id]

    def aspect_to_id(self, aspect:str) -> int:
        return self.aspect2id[aspect]

    def id_to_aspect(self, id:int) -> str:
        return self.id2aspect[id]