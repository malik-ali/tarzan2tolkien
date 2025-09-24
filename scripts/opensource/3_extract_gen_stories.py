import pandas as pd

from tqdm.auto import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from typing import List
import pandas as pd
import numpy as np
import os
from src.opensource_utils import RESP_TEMPLATE, EOS_SEQ

@dataclass
class ScriptArguments:
    gen_file: str = field(metadata={"help": "path to the generation stories"})

def extract_story(generation):
    # story is in between RESP_TEMPLATE and EOS_SEQ
    start = generation.find(RESP_TEMPLATE) + len(RESP_TEMPLATE)
    end = generation.find(EOS_SEQ)
    # return generation[:end] # TODO: remove
    return generation[start:end]

def main():
    args: ScriptArguments = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    df = pd.read_csv(args.gen_file,  dtype={'prompt': str, 'story': str, 'index': int, 'target_cefr': str, 'generation': str})
    # TODO: remove
    # df = df[~df['generation'].isna()]
    # if no column or is nan
    if 'gen_raw' not in df.columns or pd.isna(df.iloc[0]['gen_raw']):
        df['gen_raw'] = df['generation']
        df["generation"] = df["gen_raw"].map(extract_story)
        df.to_csv(args.gen_file, index=False)
    else:
        print("This file has already been processed")


if __name__ == '__main__':
    main()
