from pathlib import Path
import torch
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import TrainingArguments, Trainer

from .base_llm import BaseLLM
from .sft import test_model  # Reuse test harness

# training_dataset = "rft_filtered"
training_dataset = "rft"

def tokenize(tokenizer, question: str, reasoning: str):
    """
    Tokenizes the full reasoning (including <answer> tag) and supervises all non-padding tokens.
    """
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    full_text = f"{question.strip()} {reasoning.strip()}{tokenizer.eos_token}"
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # Supervise everything except padding
    labels = [
        token_id if attn else -100
        for token_id, attn in zip(input_ids, attention_mask)
    ]
    full["labels"] = labels
    return full

def format_example(prompt: str, answer: str) -> dict[str, str]:
    return {"question": prompt.strip(), "answer": answer.strip()}


class RFTDataset:
    def __init__(self, path=f"homework/data/{training_dataset}.json"):
        import json
        with open(path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, _, reasoning = self.data[idx]
        return question, reasoning


class TokenizedDataset:
    def __init__(self, tokenizer, data, format_fn):
        self.tokenizer = tokenizer
        self.data = data
        self.format_fn = format_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, reasoning = self.format_fn(*self.data[idx]).values()
        return tokenize(self.tokenizer, question, reasoning)


def train_model(output_dir: str = "homework/rft_model", **kwargs):
    llm = BaseLLM()

    lora_config = LoraConfig(
        r=16,  # Higher rank than SFT
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    llm.model.print_trainable_parameters()

    train_dataset = TokenizedDataset(
        llm.tokenizer,
        RFTDataset(f"homework/data/{training_dataset}.json"),
        format_example
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=20,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        report_to="tensorboard",
        logging_dir=output_dir,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)

    test_model(output_dir)


def load() -> BaseLLM:
    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(model=llm.model, model_id=str(model_path), is_trainable=False).to(llm.device)
    llm.model.eval()
    return llm


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})