"""Day 3 helper for Lab 17: prepare the training data and run a short or a full training run.

The notebook shows the real settings (the 4-bit model, the LoRA configuration, the learning rate, the batch size).
This file only does the plumbing: tokenise the examples, put them in batches, run the trainer and measure memory and time.
"""
import gc
import random
import time

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, set_seed

import edge_lab
import edge_nlu
from slm_lab import encode_chat, OUT

PAD_ID = 0          # set by train_run before training starts


def mix_and_split(main_examples, replay_examples, replay_copies=3, validation_count=100, seed=3):
    """Mix the replay examples into the main ones, shuffle, and set some aside for validation.

    Returns (train_examples, validation_examples). Replay examples are repeated a few times so they are not lost.
    """
    everything = []
    for example in main_examples:
        everything.append(example)
    for copy in range(replay_copies):
        for example in replay_examples:
            everything.append(example)
    random.Random(seed).shuffle(everything)
    validation = everything[:validation_count]
    train = everything[validation_count:]
    return train, validation


def tokenise(tok, examples, max_len=192):
    """Turn each example into token numbers. The prompt part gets label -100, so the loss only looks at the answer."""
    rows = []
    for example in examples:
        rows.append(encode_chat(tok, example["messages"], max_len))
    return rows


def collate(batch):
    """Put examples of different length into one batch: pad the short ones at the end."""
    longest = 0
    for item in batch:
        if len(item["input_ids"]) > longest:
            longest = len(item["input_ids"])
    input_ids = []
    attention = []
    labels = []
    for item in batch:
        padding = longest - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [PAD_ID] * padding)
        attention.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)
    return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention), "labels": torch.tensor(labels)}


def train_run(model, tok, rows, batch_size=8, grad_accum=1, epochs=1, max_steps=-1, learning_rate=2e-4,
              checkpointing=False, run_name="run"):
    """Train the model on the tokenised rows. Returns a dictionary: peak_gb, seconds, losses (a list), steps.

    batch_size and grad_accum: the model sees batch_size examples at a time and updates after grad_accum batches.
    checkpointing: True saves memory by recomputing some numbers instead of keeping them (a little slower).
    max_steps: a small number such as 3 makes a very short run, used only to measure memory.
    """
    global PAD_ID
    PAD_ID = tok.pad_token_id
    set_seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    arguments = TrainingArguments(
        output_dir=str(OUT / ("edge_" + run_name)), num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum, learning_rate=learning_rate,
        lr_scheduler_type="cosine", warmup_steps=5, logging_steps=10, save_strategy="no", report_to="none",
        bf16=True, gradient_checkpointing=checkpointing, remove_unused_columns=False, seed=42,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=arguments, train_dataset=Dataset.from_list(rows), data_collator=collate)
    started = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        # The setting needs more memory than the budget allows: report it instead of stopping the notebook.
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        return {"peak_gb": "out of memory", "seconds": "-", "losses": [], "steps": 0}
    seconds = time.time() - started
    losses = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            losses.append(round(entry["loss"], 4))
    return {"peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "seconds": round(seconds, 1),
            "losses": losses, "steps": trainer.state.global_step}


def count_parameters(model):
    """Returns (trainable, total): how many numbers the training changes, and how many the model has."""
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total = total + count
        if parameter.requires_grad:
            trainable = trainable + count
    return trainable, total


# ------------------------------------------------------------------------------------------------ is this GPU big enough?
MIN_GPU_GB = 12.0          # below this the lab cannot run a training: the model, the cache and the work space do not fit
MIN_BUDGET_GB = 5.5        # the training run needs about 5 GB at its busiest (measured: 4.9 GB for the largest probe)


def budget_for(total_gb, share=0.30, floor_gb=MIN_BUDGET_GB):
    """The memory budget in GB and the share of the GPU it is. Normally 30 percent of the GPU, but never below floor_gb."""
    budget = total_gb * share
    if budget < floor_gb:
        budget = floor_gb
    return round(budget, 1), round(budget / total_gb, 3)


def check_gpu():
    """Say clearly whether this GPU can run Lab 17. Stops with a plain message when it cannot."""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU was found. Lab 17 needs an NVIDIA GPU. Check that your Pod has a GPU and ask the trainer.")
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("GPU:", name, "|", round(total_gb, 1), "GB of memory")
    if total_gb < MIN_GPU_GB:
        raise RuntimeError("This GPU has " + str(round(total_gb, 1)) + " GB. Lab 17 needs at least " + str(int(MIN_GPU_GB))
                           + " GB. Ask the trainer for a bigger Pod (24 GB is what we tested).")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16, the 16-bit number format the training uses. Ask the trainer for a newer GPU (RTX 30 or 40 series, or A-series).")
    print("This GPU can run Lab 17.")
    return total_gb


def train_adapter(extra_cases, folder_name, test_cases=None, copies=8, epochs=2, learning_rate=2e-4, start_from=None):
    """Train a NEW adapter on the standard data plus your own examples (each one repeated `copies` times).

    A message that is also one of your test messages is left out, because a test must use messages the model never saw.
    start_from: the folder of an adapter you already trained. Then training CONTINUES from it, with a gentle learning rate, on your
    examples plus a small slice of the standard data (so it does not forget). Only the new patterns move; the rest stays as it was.
    Returns a dictionary: folder, seconds, peak_gb, used (how many of your examples were used).
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    test_texts = []
    if test_cases is not None:
        for case in test_cases:
            test_texts.append(case["text"].lower().strip())
    own = []
    used = 0
    for case in extra_cases:
        if case["text"].lower().strip() in test_texts:
            print("left out (it is also a test message):", case["text"])
            continue
        used = used + 1
        for number in range(copies):
            own.append(edge_nlu.make_example(case["text"], case["intent"], case["slots"]))
    main_examples = edge_nlu.make_training_examples(1200)
    if start_from is not None:
        main_examples = main_examples[:200]               # a small slice of the standard data as a reminder
        learning_rate = learning_rate / 2
    train, validation = mix_and_split(main_examples + own, edge_nlu.make_replay_examples())
    tok = AutoTokenizer.from_pretrained(edge_lab.MODEL_ID)
    four_bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(edge_lab.MODEL_ID, quantization_config=four_bit, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if start_from is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, start_from, is_trainable=True)      # continue from the adapter you trained
    else:
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    rows = tokenise(tok, train)
    result = train_run(model, tok, rows, batch_size=8, epochs=epochs, learning_rate=learning_rate, checkpointing=True,
                       run_name="mine")
    folder = str(OUT / folder_name)
    model.save_pretrained(folder)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"folder": folder, "seconds": result["seconds"], "peak_gb": result["peak_gb"], "used": used}
