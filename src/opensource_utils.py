
import os
from pprint import pprint

from datasets import Dataset
from transformers import TrainerCallback

import os
import wandb
import torch

import pandas as pd
from src.constants import OUT_DIR, DATA_DIR, TINYTOLKIEN_PATH
from src.cefr import score_to_cefr, CEFR_LEVELS


# the newline here important for the tokeniser to work
# RESP_TEMPLATE = "Story:\n"

# PROMPT_TEMPLATE = """Summary: {summary}
# CEFR: {cefr}
# """ + RESP_TEMPLATE + """{story}"""

# for llama2
RESP_TEMPLATE = "<<Story>>:\n"

PROMPT_TEMPLATE = """<<Summary>>: {summary}
<<CEFR>>: {cefr}
""" + RESP_TEMPLATE + """{story}"""

EOS_SEQ = "\n<</Story>>"


def load_tinytolkien_data():
    df = pd.read_csv(TINYTOLKIEN_PATH)
    # df = df[df['split'] == split & df['target_cefr'] == cefr_level]
    df['story'] = df['generation']
    df['cefr'] = df['target_cefr']
    df['summary'] = df['prompt']
    return df[['summary', 'story', 'cefr', 'split']]


def format_text(example, include_eos=True):
    story = example["story"]
    cefr = example["cefr"]
    summary = example["summary"]
    tt = PROMPT_TEMPLATE.format(summary=summary, cefr=cefr, story=story)
    if include_eos:
        tt += EOS_SEQ
    # return tt
    return {"input": tt}

def format_text_to_gen(example):
    cefr = example["cefr"]
    summary = example["summary"]
    tt = PROMPT_TEMPLATE.format(summary=summary, cefr=cefr, story="")
    # return tt
    return {"input": tt}


def format_text_batch(examples):
    output_texts = []
    for i in range(len(examples['prompt'])):
        text = examples['response'][i]
        prompt = examples['prompt'][i]
        tt = PROMPT_TEMPLATE.format(prompt=prompt, response=text)
        output_texts.append(tt)
    return output_texts


class GenSentenceCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # check if main process in ddp training
        if not state.is_world_process_zero:
            return
        # get model and have it generate some sentences from the train and validation set
        if not state.is_world_process_zero:
            return
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        # valid_dataset = kwargs["eval_dataset"]
        df = load_tinytolkien_data()

        df["true_resp"]= df["story"]
        df["story"] = ""
        train_data = Dataset.from_pandas(df[df['split'] == 'train'], split='train', preserve_index=False)
        valid_data = Dataset.from_pandas(df[df['split'] == 'test'], split='validation', preserve_index=False)

        # print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

        train_data = train_data.map(format_text_to_gen)
        valid_data = valid_data.map(format_text_to_gen)

        train_dataset = train_data
        valid_dataset = valid_data

        # print(model.device)
        text_table = wandb.Table(columns=["epoch", "split", "prompt", "response", "true_resp"])
        N = 2
        with torch.no_grad():
            # generate sentences from the train set
            train_sentences = []
            for i in range(N):
                prompt = train_dataset[i]["input"]

                # the -1 to remove the eos_token added by the tokenizer
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][:, :-1].to(model.device)
                train_sentences.append(model.generate(
                    input_ids=input_ids,
                    max_length=500,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True
                ))

            # generate sentences from the validation set
            valid_sentences = []
            for i in range(N):
                prompt = valid_dataset[i]["input"]
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][:, :-1].to(model.device)
                valid_sentences.append(model.generate(
                    input_ids=input_ids,
                    max_length=500,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True
                ))

            # print the generated sentences
            print("Generating sentences...")
            for i in range(N):
                prompt = train_dataset[i]["input"]
                tru_resp = train_dataset[i]["true_resp"]
                sent = tokenizer.decode(train_sentences[i]["sequences"][0])
                text_table.add_data(state.epoch, "train", prompt, sent, tru_resp)
            # print("Generated sentences from the validation set:")
            for i in range(N):
                prompt = valid_dataset[i]["input"]
                tru_resp = valid_dataset[i]["true_resp"]
                sent = tokenizer.decode(valid_sentences[i]["sequences"][0])
                text_table.add_data(state.epoch, "val", prompt, sent, tru_resp)

        wandb.log({"generated_sentences": text_table})
