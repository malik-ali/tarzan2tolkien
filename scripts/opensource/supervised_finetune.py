import os
from pprint import pprint

from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed, HfArgumentParser

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from src.opensource_utils import RESP_TEMPLATE, GenSentenceCallback, load_tinytolkien_data, format_text
from typing import Optional
from dataclasses import dataclass, field
from src.constants import OUT_DIR


from transformers import AutoModelForCausalLM


"""
Fine-Tune
"""
@dataclass
class ScriptArguments:
    model_name: str = field(metadata={"help": "the model name"})
    # tokenizer_name: str = field(metadata={"help": "the tokenizer name"})

    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    weight_decay: Optional[float] = field(default=1e-5, metadata={"help": "the weight decay"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})

    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "resume from a checkpoint"})

    train_steps: Optional[int] = field(default=10000, metadata={"help": "total number of training steps"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "log every n steps"})
    eval_freq: Optional[int] = field(default=10, metadata={"help": "evaluate every n steps"})

    seed: Optional[int] = field(default=42, metadata={"help": "seed for reproducibility"})

    wandb_name: Optional[str] = field(default=None, metadata={"help": "name for this wandb run"})
    notes: Optional[str] = field(default=None, metadata={"help": "some notes for thisrun"})

    save_freq: Optional[int] = field(default=20, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default=os.path.join(OUT_DIR, 'tinystories', 'opensource', 'finetuned', 'default'), metadata={"help": "where to save the model"})

def parse_args():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    return script_args


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def create_datasets(tokenizer, args):
    df = load_tinytolkien_data()

    train_data = Dataset.from_pandas(df[df['split'] == 'train'], split='train', preserve_index=False)
    valid_data = Dataset.from_pandas(df[df['split'] == 'test'], split='validation', preserve_index=False)

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    train_data = train_data.map(format_text)
    valid_data = valid_data.map(format_text)

    train_dataset = train_data
    valid_dataset = valid_data

    return train_dataset, valid_dataset


def run_training(args, tokenizer, train_data, val_data):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # TODO: some tokenizers need to add context to the response template so that it can match
    # how it will be tokenized in the dataset. For example we will sometimes need to add a \n at the beginning of the response template
    RESP_TEMPLATE_WITH_CONTEXT = "\n" + RESP_TEMPLATE
    response_template_ids = tokenizer.encode(RESP_TEMPLATE_WITH_CONTEXT, add_special_tokens=False)[2:]


    collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,
                                               response_template=response_template_ids)
    train_data.start_iteration = 0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=500,
        evaluation_strategy="steps",
        max_steps=args.train_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_checkpointing=False,
        weight_decay=args.weight_decay,
        report_to="wandb",
        run_name=args.wandb_name,
        ddp_find_unused_parameters=False,
    )

    current_device = Accelerator().local_process_index

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        trust_remote_code=True,
        device_map={"": current_device},
        use_auth_token=True
    )

    model.config.use_cache = False

    print(model)

    # This is super slow but can be used to debug:
    # callbacks = [GenSentenceCallback()]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="input",
        data_collator=collator,
        max_seq_length=4096,
        eval_dataset=val_data,
        peft_config=lora_config,
        #callbacks=callbacks
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_eos_token=True, trust_remote_code=True)
    # sets pad to ▁***
    tokenizer.pad_token_id = 18610
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    run_training(args, tokenizer, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = parse_args()
    assert args.model_name != "", "Please provide the model path"

    set_seed(args.seed)

    logging.set_verbosity_error()
    pprint(args)

    main(args)
