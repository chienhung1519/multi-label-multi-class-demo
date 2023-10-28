from argparse import ArgumentParser, Namespace
import pandas as pd
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
    data = pd.read_csv(args.data_path)
    data.columns = Path(args.src_dir, "column_names.txt").read_text().splitlines()

    # Generate K-fold cross validation data
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train = data.iloc[train_index]
        test =  data.iloc[test_index]
        eval = train.sample(n=len(test)//2, random_state=args.seed)
        train = train.drop(eval.index)

        data_dir = Path(args.data_path).parents[0] / "processed" / f"fold_{i+1}"
        data_dir.mkdir(parents=True, exist_ok=True)
        train.to_json(data_dir / "/train.json", orient="records", indent=2)
        eval.to_json(data_dir / "/eval.json", orient="records", indent=2)
        test.to_json(data_dir / "/test.json", orient="records", indent=2)

    # Store base output_dir
    base_output_dir = args.output_dir

    # Start K-fold cross validation
    for i in range(args.n_splits):
        args.train_data_path = Path(args.data_path).parents[0] / "processed" / f"fold_{i+1}" / "train.json"
        args.eval_data_path = Path(args.data_path).parents[0] / "processed" / f"fold_{i+1}" / "eval.json"
        args.test_data_path = Path(args.data_path).parents[0] / "processed" / f"fold_{i+1}" / "test.json"
        args.output_dir = Path(args.output_dir) / f"fold_{i+1}"

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
            save_steps=500,
            save_total_limit=1,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            overwrite_output_dir=True,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_module.train_dataset(),
            eval_dataset=dataset_module.eval_dataset(),
            compute_metrics=compute_metrics,
        )

        # Training
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset_module.train_dataset())
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Test
        predictions = trainer.predict(dataset_module.test_dataset())
        y_predict = [dataset_module.id_to_label(p) for p in predictions.predictions.argmax(axis=-1)]
        outputs = pd.concat(
            [dataset_module.test_data, pd.DataFrame(y_predict, columns=["predictions"])], axis=1
        )
        outputs.to_excel(args.output_dir+"/outputs.xlsx", index=False)

    # Classification report
    all_outputs = pd.DataFrame()
    for i in range(args.n_splits):
        outputs = pd.read_excel(Path(args.output_dir) / f"fold_{i+1}" / "outputs.xlsx")
        all_outputs = pd.concat([all_outputs, outputs], axis=0)
    results = {}
    for aspect_name in dataset_module.aspect_names:
        aspect_outputs = outputs[outputs["aspect_ids"]==aspect_name]
        results[aspect_name] = classification_report(aspect_outputs["labels"], aspect_outputs["predictions"], output_dict=True, digits=4)
    results = pd.DataFrame(results).transpose()
    results.to_excel(base_output_dir + "/classification_report.xlsx")
            
if __name__ == "__main__":
    main()