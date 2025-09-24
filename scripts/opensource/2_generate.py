import pandas as pd

from tqdm.auto import tqdm
from dataclasses import dataclass, field
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from typing import Optional
import os
import time
from peft import AutoPeftModelForCausalLM


from typing import List
import torch
import pandas as pd
import numpy as np
import os
from accelerate import Accelerator
from src.opensource_utils import format_text_to_gen

@dataclass
class ScriptArguments:
    to_gen: str = field(metadata={"help": "path to the generation requests"})
    save_every: Optional[int] = field(default=2, metadata={"help": "save every n generations"})
    model_dir: Optional[str] = field(default=None, metadata={"help": "model to use for generation"})
    is_local_peft: Optional[bool] = field(default=False, metadata={"help": "whether the model is a local peft model"})


def generate_at_level(summary, target_cefr, model, tokenizer):
    d = dict(summary=summary, cefr=target_cefr, story="")
    prompt = format_text_to_gen(d)["input"]


    with torch.no_grad():
        # no need of -1 to remove the eos_token by the tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            max_length=2048,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
            # return_dict_in_generate=True
        )

        gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(prompt)
        print(gen)

        print()
        return gen

def load_model(model_dir, is_local_peft=False):
    device_map = {"": Accelerator().local_process_index}
    if "mistral" in model_dir:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", add_eos_token=False, trust_remote_code=True, use_fast = True)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id =  tokenizer.unk_token_id
        tokenizer.padding_side = 'left'
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if is_local_peft:
        model = AutoPeftModelForCausalLM.from_pretrained(model_dir, device_map=device_map, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map, torch_dtype=torch.bfloat16)

    model.eval()
    return model, tokenizer

def main():
    args: ScriptArguments = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]

    model, tokenizer = load_model(args.model_dir, args.is_local_peft)

    df = pd.read_csv(args.to_gen,  dtype={'prompt': str, 'index': int, 'target_cefr': str, 'generation': str})
    if 'generation' not in df.columns:
        df['generation'] = pd.NA


    tqdm.pandas()

    cnt = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if not pd.isnull(row['generation']):
            continue

        gen = generate_at_level(row['prompt'], row['target_cefr'], model, tokenizer)
        cnt += 1
        df.loc[i, 'generation'] = gen
        if cnt % args.save_every == 0:
            print('saving...')
            df.to_csv(args.to_gen, index=False)


        # time.sleep(1) # avoid rate limit

    df.to_csv(args.to_gen, index=False)


if __name__ == '__main__':
    main()
