from argparse import ArgumentParser, Namespace
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from transformers import TrainingArguments, Trainer, AutoTokenizer, set_seed

from dataset_module import MultiLabelMultiClassDataset
from model_module import MultiLabelMultiClassModel, compute_metrics

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--src_dir", type=str, default="./src")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    # Load arguments
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load data
    column_names = Path(args.src_dir, "column_names.txt").read_text().splitlines()
    data = pd.read_csv(args.data_path, sep="\t", names=column_names, encoding="utf-8")

    # Generate K-fold cross validation data
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        # Create data directory
        data_dir = Path(args.data_path).parents[1] / "processed" / f"fold_{i+1}"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Skip if data already exists
        if (data_dir / "train.json").exists() and (data_dir / "eval.json").exists() and (data_dir / "test.json").exists():
            continue
        # Split data
        train = data.iloc[train_index]
        test =  data.iloc[test_index]
        eval = train.sample(n=len(test)//2, random_state=args.seed)
        train = train.drop(eval.index)
        # Save data
        train.to_json(data_dir / "train.json", orient="records", indent=2)
        eval.to_json(data_dir / "eval.json", orient="records", indent=2)
        test.to_json(data_dir / "test.json", orient="records", indent=2)

    # Store base output_dir
    base_output_dir = args.output_dir

    # Start K-fold cross validation
    for i in range(args.n_splits):
        args.train_data_path = Path(args.data_path).parents[1] / "processed" / f"fold_{i+1}" / "train.json"
        args.eval_data_path = Path(args.data_path).parents[1] / "processed" / f"fold_{i+1}" / "eval.json"
        args.test_data_path = Path(args.data_path).parents[1] / "processed" / f"fold_{i+1}" / "test.json"
        args.output_dir = base_output_dir + "/" + f"fold_{i+1}"

        # Load datasets
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        dataset_module = MultiLabelMultiClassDataset(tokenizer, args)

        # Load model
        model = MultiLabelMultiClassModel(
            encoder_name_or_path=args.model_name_or_path,
            num_labels=args.num_labels,
            aspect_ids=dataset_module.aspect_ids,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            seed=args.seed,
            gradient_accumulation_steps=args.accumulate_grad_batches,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_epsilon=args.adam_epsilon,
            num_train_epochs=args.epochs,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            fp16=True,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            overwrite_output_dir=True,
            report_to="none",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_module.train_dataset,
            eval_dataset=dataset_module.eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Training
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset_module.train_dataset)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Test
        predictions = trainer.predict(dataset_module.test_dataset)
        preds = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
        y_predict = [dataset_module.id_to_label(p) for p in np.argmax(preds, axis=1)]
        outputs = pd.concat(
            [dataset_module.test_data, pd.DataFrame(y_predict, columns=["predictions"])], axis=1
        )
        outputs["aspect"] = outputs["aspect_ids"].apply(dataset_module.id_to_aspect)
        outputs["labels"] = outputs["labels"].apply(dataset_module.id_to_label)
        outputs = outputs[["ReviewTitle", "ReviewText", "aspect", "labels", "predictions"]]
        outputs.to_excel(args.output_dir+"/outputs.xlsx", index=False)

    # Classification report
    all_outputs = pd.DataFrame()
    for i in range(args.n_splits):
        outputs = pd.read_excel(Path(base_output_dir) / f"fold_{i+1}" / "outputs.xlsx")
        all_outputs = pd.concat([all_outputs, outputs], axis=0)
    results = []
    for aspect_name in dataset_module.aspect_names:
        aspect_outputs = outputs[outputs["aspect"]==aspect_name]
        cr = classification_report(aspect_outputs["labels"], aspect_outputs["predictions"], output_dict=True, digits=4)
        aspect_results = []
        label_list = ["Negative", "Positive", "NoFillingIn"]
        metric_list = ["precision", "recall", "f1-score"]
        for label in label_list:
            aspect_results.extend([cr[label]["precision"], cr[label]["recall"], cr[label]["f1-score"]])
        results.append([aspect_name] + aspect_results)
    results = pd.DataFrame(results)
    results.set_index(0, inplace=True)
    results.columns = pd.MultiIndex.from_tuples([(label, metric) for label in label_list for metric in metric_list])
    results.to_excel(base_output_dir + "/classification_report.xlsx")
            
if __name__ == "__main__":
    main()