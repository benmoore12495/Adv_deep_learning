from .base_llm import BaseLLM
from .data import Dataset, benchmark

from pathlib import Path
default_output_dir = Path(__file__).parent / "sft_model"


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
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)
    input_ids = full["input_ids"]

    # Find the start of <answer> tag in the *text*, then count tokens up to that
    answer_start = full_text.find("<answer>")
    if answer_start == -1:
        raise ValueError("No <answer> tag found!")

    answer_token_start = len(tokenizer(full_text[:answer_start])["input_ids"])
    labels = [-100] * answer_token_start + input_ids[answer_token_start:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    return {
        # "question": prompt.strip(),  # No Q: or A:
        "question": prompt,  # No Q: or A:
        "answer": f"<answer>{round(float(answer), 2)}</answer>"
    }




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

    # def __getitem__(self, idx):
    #     formated_data = self.format_fn(*self.data[idx])
    #     return tokenize(self.tokenizer, **formated_data)
    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        tokenized = tokenize(self.tokenizer, **formated_data)

        if idx == 0:
            print("\n--- Tokenization Debug ---")
            print("Original question:", formated_data["question"])
            print("Input IDs:", tokenized["input_ids"])
            print("Tokens:", self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"]))
            print("Labels:", tokenized["labels"])

        return tokenized


def train_model(
    output_dir: str = str(default_output_dir),
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


def test_model(ckpt_path: str):
    from pathlib import Path
    from peft import PeftModel

    testset = Dataset("valid")
    llm = BaseLLM()

    adapter_path = Path(ckpt_path)
    llm.model = PeftModel.from_pretrained(model=llm.model, model_id=str(adapter_path), is_trainable=False).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

    print("\n--- Sample Generations ---")
    for i in range(5):
        question, true_answer = testset[i]
        # raw_output = llm.generate(question)
        formatted_prompt = f"Q: {question}\nA: <answer>"
        print("\n[DEBUG] Prompt given to LLM:\n", formatted_prompt)
        raw_output = llm.generate(formatted_prompt)
        parsed = llm.parse_answer(raw_output)

        print(f"\nQ{i+1}: {question}")
        print("Raw Generation:", repr(raw_output))
        print("Parsed Answer:", parsed)
        print("Expected:", true_answer)



if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
