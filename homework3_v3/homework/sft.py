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
    """
    Tokenizes the prompt (question) and supervises only the answer portion.
    - No chat formatting!
    - Only the answer part (<answer>...</answer>) is used for loss computation.
    """
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    # Format the full string
    full_text = f"{question.strip()} {answer.strip()}{tokenizer.eos_token}"

    # Tokenize full string
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # Identify where answer starts
    q_ids = tokenizer(question.strip(), add_special_tokens=False)["input_ids"]
    q_len = len(q_ids)

    # Mask labels before the answer
    labels = [-100] * q_len + input_ids[q_len:]

    # Also mask out padding tokens
    labels = [
        label if attn else -100
        for label, attn in zip(labels, attention_mask)
    ]

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    chat_prompt = (
        "<|im_start|>system\nYou are a helpful unit conversion assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {
        "question": chat_prompt,
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

    def __getitem__(self, idx):
        formatted_data = self.format_fn(*self.data[idx])
        question = formatted_data["question"]
        answer = formatted_data["answer"]

        # Combine the full string like we do in `tokenize()`
        full_text = f"{question} {answer}"

        tokenized = tokenize(self.tokenizer, question, answer)

        # if idx < 3:  # print first few samples only
        #     print("\n--- Tokenization Debug ---")
        #     print("Full training input (pre-tokenization):", repr(full_text))
        #     print("Input IDs:", tokenized["input_ids"])
        #     print("Tokens:", self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"]))
        #     print("Labels:", tokenized["labels"])
        #     print("--- End Debug ---")

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
        # num_train_epochs=5,
        # num_train_epochs=10, ## 46% accuracy 
        # num_train_epochs=15, ## 56% accuracy 
        num_train_epochs=20, ## 62% accuracy 
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
    for i in range(10):
        question, true_answer = testset[i]
        # raw_output = llm.generate(question)
        chat_prompt = llm.format_prompt(question)
        print("\n[DEBUG] Prompt given to LLM:\n", chat_prompt)
        raw_output = llm.generate(chat_prompt)
        parsed = llm.parse_answer(raw_output)

        print(f"\nQ{i+1}: {question}")
        print("Raw Generation:", repr(raw_output))
        print("Parsed Answer:", parsed)
        print("Expected:", true_answer)

if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
