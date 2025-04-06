from .base_llm import BaseLLM
from .data import Dataset, benchmark

# def load() -> BaseLLM:
#     from pathlib import Path
#     from peft import PeftModel

#     model_name = "sft_model"
#     model_path = Path(__file__).parent / model_name

#     llm = BaseLLM()
#     # llm.model = PeftModel.from_pretrained(model_path, model=llm.model, is_trainable=False).to(llm.device)
#     llm.model = PeftModel.from_pretrained(model=llm.model, model_id=model_path, is_trainable=False).to(llm.device)

#     llm.model.eval()

#     return llm

def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(model=llm.model, model_id=str(model_path), is_trainable=False).to(llm.device)
    llm.model.eval()
    return llm



def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # raise NotImplementedError()
    formatted_answer = f"<answer>{round(float(answer), 2)}</answer>"
    return {"question": prompt, "answer": formatted_answer}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    # raise NotImplementedError()
    # test_model(output_dir)

    import torch
    from transformers import TrainingArguments, Trainer
    from peft import get_peft_model, LoraConfig, TaskType
    from pathlib import Path
    from .base_llm import BaseLLM
    from .data import Dataset

    # 1. Load base model
    llm = BaseLLM()

    # 2. Add LoRA adapter
    lora_config = LoraConfig(
        r=8,  # adjust so the final adapter stays under 20MB
        lora_alpha=32,  # typically 4-5x the rank
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.print_trainable_parameters()

    # 3. Enable grads if using gradient checkpointing
    llm.model.enable_input_require_grads()

    # 4. Prepare dataset
    train_dataset = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)

    # 5. Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_dir=output_dir,
        report_to="tensorboard",
        save_strategy="epoch",
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),  # Mixed precision
    )

    # 6. Trainer setup
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 7. Train
    trainer.train()

    # 8. Save only adapter
    # trainer.save_model(Path(__file__).parent / "sft_model")
    trainer.save_model(output_dir)


    # 9. Optional test
    test_model(output_dir)


# def test_model(ckpt_path: str):
#     from pathlib import Path
#     from peft import PeftModel

#     testset = Dataset("valid")
#     llm = BaseLLM()

#     adapter_path = Path(ckpt_path)
#     # llm.model = PeftModel.from_pretrained(adapter_path, model=llm.model, is_trainable=False).to(llm.device)
#     llm.model = PeftModel.from_pretrained(model=llm.model, model_id=adapter_path, is_trainable=False).to(llm.device)

#     benchmark_result = benchmark(llm, testset, 100)
#     print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

def test_model(ckpt_path: str):
    from pathlib import Path
    from peft import PeftModel

    testset = Dataset("valid")
    llm = BaseLLM()

    adapter_path = Path(ckpt_path)
    llm.model = PeftModel.from_pretrained(model=llm.model, model_id=str(adapter_path), is_trainable=False).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")



if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
