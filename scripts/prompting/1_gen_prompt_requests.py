import pandas as pd

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional
from src.cefr import CEFR_LEVELS
import os


@dataclass
class ScriptArguments:
    prompt_data: str = field(metadata={"help": "path to a csv of text data with prompts"})
    outfile: str = field(metadata={"help": "path to save the generated requests"})
    n_per_cefr: Optional[int] = field(default=10, metadata={"help": "number of samples per cefr level"})
    # target_cefr: Optional[str] = field(default=None, metadata={"help": "target cefr level"})


def main():
    args: ScriptArguments = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    if os.path.exists(args.outfile):
        raise ValueError(f'File {args.outfile} already exists. Please delete it or choose a different file name.')

    # Load the data
    df = pd.read_csv(args.prompt_data)

    df_sub = df
    df_sub['index'] = df_sub.index
    to_gen = CEFR_LEVELS
    df_sub = df_sub.loc[df_sub.index.repeat(args.n_per_cefr)].reset_index(drop=True)
    df_sub['target_cefr'] = df_sub['index'].apply(lambda x: to_gen)
    df_sub['prompt'] = df_sub['summary'].apply(lambda x: x.strip())
    df_sub['story'] = df_sub['story'].apply(lambda x: x.strip())
    df_sub.drop(columns=['summary'], inplace=True)
    df_sub = df_sub.explode('target_cefr')
    df_sub['generation'] = None
    df_sub['gpt_request'] = None
    print(df_sub)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    df_sub.to_csv(args.outfile, index=False)


if __name__ == '__main__':
    # main_uncond()
    main()
