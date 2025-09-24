import pandas as pd

from tqdm.auto import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional
from src.cefr import CEFR_LEVELS
import os
import time
import json
from src.promptify import CEFRControlPrompt
from src.constants import OUT_DIR

from typing import List

import pandas as pd
import numpy as np

SYSTEM_PROMPT = """You are an essay grader that provides grades to a writing task. In this task, the student needs to write a short story according to a given plot summary."""

USER_PROMPT = """
Grade the student's story terms of the following criteria:
    - language: The correctness of the grammar and language. Graded on a scale of 1 to 10
    - consistency: Consistency with the given story plot. Graded on a scale of 1 to 10
    - proficiency: a score measuring the proficiency of the student's writing, as measured by the CEFR scale. Graded as one of A1, A2, B1, B2, C1, C2

Ouptput the grade as a single json dictionary e.g. {{"language": 8, "consistency": 7, "proficiency": "B2"}}
## Plot summary:
{prompt}

## Generated story:
{story}
"""


@dataclass
class ScriptArguments:
    input_file: str = field(metadata={"help": "path to the file with text to score."})
    output_file: str = field(metadata={"help": "path to the file to save scores."})
    gen_column: str = field(metadata={"help": "column to score."})
    prompt_column: str = field(metadata={"help": "column to use as reference."})

    save_every: Optional[int] = field(default=2, metadata={"help": "save every n generations"})
    model: Optional[str] = field(default="gpt-4", metadata={"help": "model to use for scoring"})


import os
import openai

def call_gpt(system_prompt: str,
             user_prompt: str,
             model: str):
    """

    :param system_prompt: the main direction to the system
    :param user_prompt: what you actually want a response to
    :param example_inputs: A list of example inputs that the user might provide to the system. These will have role "USER" in the chat interface. This is optional.
    :param example_responses: A list of example responses that the system might provide to the user. These will have role "ASSISTANT" in the chat interface. This is optional.
    :param version: The version of the model to use. This is optional.
    """
    # system_prompt = SYSTEM_PROMPT
    # user_prompt = USER_PROMPT.format(prompt = prompt, story=story)
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8,
        max_tokens = 512,
    )
    resp = completion.choices[0].message["content"]
    #print(resp)

    return resp


def score_generation(prompt, generation, model):
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT.format(prompt=prompt, story=generation)

    combined = f"====== System ======\n{system_prompt}\n====== User ======\n{user_prompt}"

    try:
        # print('*'*100)
        # print(system_prompt)
        # print('~~~~~~~~~~~~~~~')
        # print(user_prompt)
        # print('===================')
        gen = call_gpt(system_prompt, user_prompt, model)

        grade = json.loads(gen)
        # print(target_cefr)
        return grade, combined

    except json.JSONDecodeError as e:
        print("Error: ", e)
        return None, f'<JSONERROR>:{str(e)}' + '\n\n' + gen
    except Exception as e:
        print("Error: ", e)
        return None, f'<ERROR>:{str(e)}' + '\n\n' + combined

def main():
    from pprint import pprint
    args: ScriptArguments = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    model = args.model


    if not os.path.exists(args.output_file):
        fn = args.input_file
    else:
        fn = args.output_file


    df = pd.read_csv(fn) # dtype={'prompt': str, 'story': str, 'index': int, 'target_cefr': str, 'generation': str})
    # sample one row per index, target_cefr. This is for cases where we have multiple generations for the same prompt
    df = df.groupby(['index', 'target_cefr']).apply(lambda x: x.sample(1)).reset_index(drop=True)

    if not 'score_language' in df.columns:
        df['score_language'] = np.nan
        df['score_consistency'] = np.nan
        df['score_proficiency'] = None

    tqdm.pandas()


    cnt = 1
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if not pd.isnull(row['score_consistency']):
            continue

        grade, _ = score_generation(row[args.prompt_column], row[args.gen_column], model)
        if grade is None:
            df.loc[i, f'error'] = True
            continue
        for k, v in grade.items():
            df.loc[i, f'score_{k}'] = v

        if cnt % args.save_every == 0:
            print('saving...')
            df.to_csv(args.output_file, index=False)

        cnt += 1

        # time.sleep(1) # avoid rate limit

    df.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
