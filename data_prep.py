import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def main(raw_dataset_path, tokenized_dataset_path):
    # Step 1: Download and save raw dataset
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        num_proc=24
    )
    ds.save_to_disk(raw_dataset_path)

    # Step 2: Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    tokenized_ds = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Step 3: Save tokenized dataset
    tokenized_ds.save_to_disk(tokenized_dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and tokenize smollm-corpus cosmopedia-v2")
    parser.add_argument("--raw_dataset_path", required=True, help="Path to save the raw dataset")
    parser.add_argument("--tokenized_dataset_path", required=True, help="Path to save the tokenized dataset")
    args = parser.parse_args()

    main(args.raw_dataset_path, args.tokenized_dataset_path)