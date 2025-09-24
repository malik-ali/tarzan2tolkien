from dataclasses import dataclass, field
import pickle
from typing import Optional

from datasets import Dataset
import torch
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline,BitsAndBytesConfig

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed, create_reference_model
from trl.core import LengthSampler


from src.cefr import cefr_to_score, CEFR_LEVELS
import numpy as np
from random import choices
import os
from src.cefr_scorer.featurise import predict
from src.constants import MODEL_DIR

from src.opensource_utils import format_text_to_gen
from src.opensource_utils import load_tinytolkien_data, format_text_to_gen



tqdm.pandas()

import os



@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    model_name: str = field(metadata={"help": "the model name"})

    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    steps: Optional[int] = field(default=500, metadata={"help": "number of training steps (epochs = steps/bs)"})

    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})

    lora_r: Optional[int] = field(default=16, metadata={"help": "lora r"})
    lora_alpha: Optional[int] = field(default=32, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})

    wandb_project: Optional[str] = field(default="llama2_rlhf", metadata={"help": "name of wandb project to log to"})
    wandb_group: Optional[str] = field(default=None, metadata={"help": "experiment group in wandb project"})
    wandb_name: Optional[str] = field(default=None, metadata={"help": "name for this wandb run"})
    notes: Optional[str] = field(default=None, metadata={"help": "some notes for this run"})

    save_freq: Optional[int] = field(default=5, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "where to save the model"})

    seed: Optional[int] = field(default=42, metadata={"help": "seed for reproducibility"})


def load_model(model_name):
    current_device = Accelerator().local_process_index

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False, trust_remote_code=True)
    print("Pad ID: ", tokenizer.pad_token_id)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
       # target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        #load_in_8bit=True,
        trust_remote_code=True,
        device_map={"": current_device},
        use_auth_token=True,
        peft_config=lora_config
    )

    return model, tokenizer

def load_dataset():
    df = load_tinytolkien_data()
    train_dataset = Dataset.from_pandas(df[df['split'] == 'train'], split='train', preserve_index=False)

    print(f"Size of the train set: {len(train_dataset)}. ")

    return train_dataset

def load_ppo_config(args):
    seed = args.seed
    np.random.seed(seed)
    set_seed(seed)

    tracker_kwargs = {
        "wandb": dict(
            group=args.wandb_group,
            name=args.wandb_name,
            notes=args.notes
        )
    }
    ppo_config = PPOConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_score_scaling=True,
        use_score_norm=True,
        remove_unused_columns=False,
        init_kl_coef=0.35,
        seed=seed,
        log_with="wandb",
        tracker_project_name=args.wandb_project,
        tracker_kwargs=tracker_kwargs
    )
    return ppo_config

def cefr_to_score(cefr):
    return CEFR_LEVELS.index(cefr)

def reward_model(sent_scorer, generated, target_cefrs, safe_print=print):
    targets = np.array([cefr_to_score(c) for c in target_cefrs])
    y, _ = predict(sent_scorer['model'], sent_scorer['scaler'], sent_scorer['feats'],
            generated)

    MIN_CEFR_PRED = -0.5
    MAX_CEFR_PRED = 5.5
    preds = np.array(y)
    preds = np.clip(preds, MIN_CEFR_PRED, MAX_CEFR_PRED)

    for s,t,p in zip(generated[:2], targets[:2], preds[:2]):
        safe_print(f'(t:{t:.2f}, p:{p:.2f})', s)

    ret = -(preds - targets)**2
    return ret

def main(args):
    model, tokenizer = load_model(args.model_name)
    data = load_dataset()
    ppo_config = load_ppo_config(args)
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=data,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0,
        "top_p": 1,
        "do_sample": True,
        "max_new_tokens": args.output_max_length,
        "pad_token_id": tokenizer.pad_token_id,
    }

    cefr_model_store = pickle.load(open(os.path.join(MODEL_DIR, 'kaggle_linreg_en_cefr_score.b'), 'rb'))

    safe_print = ppo_trainer.accelerator.print
    device = Accelerator().local_process_index
    step = 0
    for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
        for i, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            summary_batch = batch["summary"]
            # choose a random cefr level for each story
            # cefrs = choices(CEFR_LEVELS, k=len(summary_batch))
            # assign cefrs sequentially to stories
            #to_gen = CEFR_LEVELS.index("C2")
            #cefrs = [CEFR_LEVELS[to_gen] for i in range(len(summary_batch))]

            cefrs = [CEFR_LEVELS[i % len(CEFR_LEVELS)] for i in range(len(summary_batch))]

            queryOrig = [format_text_to_gen(dict(summary=summary, cefr=cefr))["input"] for cefr, summary in zip(cefrs, summary_batch)]
            query_tensors = [tokenizer(q, return_tensors="pt")["input_ids"].to(device).squeeze() for q in queryOrig]
            batch["query"] = [tokenizer.decode(q) for q in query_tensors]

            #### Get response from SFTModel
            safe_print("generating...")
            bs = min(args.batch_size, 32)
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, batch_size=bs, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

            #### Compute reward score
            reward_outputs = reward_model(cefr_model_store, batch["response"], cefrs, safe_print=safe_print)
            rewards = [torch.tensor(output) for output in reward_outputs]

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            step += 1

            for lvl in CEFR_LEVELS:
                key = "env/reward_" + lvl
                dd = [r.cpu().numpy() for r, t in zip(rewards, cefrs) if t == lvl]
                if dd:
                    stats[key] = np.mean(dd)

            ppo_trainer.log_stats(stats, batch, rewards)
            safe_print(f"Step: {step}")

            if ppo_trainer.accelerator.is_local_main_process and args.save_freq and step % args.save_freq == 0:
                safe_print("saving...")
                ppo_trainer.save_pretrained(os.path.join(args.output_dir, f"step_{step}"))




if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
