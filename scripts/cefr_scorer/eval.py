import pickle
import pandas as pd

from tqdm.auto import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional, List
import os
import time

import pandas as pd
import numpy as np

from src.constants import MODEL_DIR


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    input_file: str = field(metadata={"help": "path to the file with text to score."})
    output_file: str = field(metadata={"help": "path to save the scored file."})
    # optional list of columns to score from the input file
    model: str = field(default='kaggle_edia_bal', metadata={"help": "model to use for scoring."})
    columns: Optional[List[str]] = field(default_factory=lambda: ['generation'], metadata={"help": "columns to score from the input file."})

def main():
    args: ScriptArguments = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    print(args.columns)

    df = pd.read_csv(args.input_file)
    df.dropna(subset=args.columns, inplace=True)

    model = args.model
    sent_scorer = pickle.load(open(os.path.join(MODEL_DIR, f'linreg_en_{model}.b'), 'rb'))

    from src.cefr_scorer.featurise import predict

    for col in args.columns:
        text_batch = [a for a in df[col]]
        print('Processing: ', col)
        print(len(text_batch))
        y, _ = predict(sent_scorer['model'], sent_scorer['scaler'], sent_scorer['feats'], [str(sent) for sent in text_batch])
        df[f'{col}_{model}_ypred'] = y


    df.to_csv(args.output_file, index=False)







if __name__ == '__main__':
    main()