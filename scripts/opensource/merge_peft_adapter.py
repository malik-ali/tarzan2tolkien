from dataclasses import dataclass, field

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, HfArgumentParser
from accelerate import Accelerator


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    checkpoint_dir: str = field(metadata={"help": "the directory of the peft model"})
    output_name: str = field(metadata={"help": "the model name"})
    tokenizer_name: str = field(metadata={"help": "the tokenizer name"})


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]
valid_toks = {"meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"}

if args.tokenizer_name == "meta-llama/Llama-2-7b-hf":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_eos_token=False, trust_remote_code=True)
    tokenizer.pad_token_id = 18610
    # sets pad to ▁***
elif args.tokenizer_name == "mistralai/Mistral-7B-v0.1":
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", add_eos_token=False, trust_remote_code=True, use_fast = True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id =  tokenizer.unk_token_id
    tokenizer.padding_side = 'left'
else:
    raise ValueError(f"Tokenizer name must be one of {valid_toks}")


device_map = {"": Accelerator().local_process_index}
model = AutoPeftModelForCausalLM.from_pretrained(args.checkpoint_dir, device_map=device_map, torch_dtype=torch.bfloat16)

model.eval()
model = model.merge_and_unload()

model.save_pretrained(f"{args.output_name}")
tokenizer.save_pretrained(f"{args.output_name}")
model.push_to_hub(f"{args.output_name}", use_temp_dir=False)
tokenizer.push_to_hub(f"{args.output_name}", use_temp_dir=False)
