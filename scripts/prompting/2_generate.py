import pandas as pd

from tqdm.auto import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional
import os
import openai

from src.promptify import CEFRControlPrompt


import pandas as pd

SYSTEM_PROMPT =f"""You are a writer that generates a story according to a given plot summary."""

USER_PROMPT = """
Write a short story (3-5 paragraphs) with the following plot. Output the story only and no other text!

Plot: {prompt}

"""


@dataclass
class ScriptArguments:
    to_gen: str = field(metadata={"help": "path to the generation requests"})
    save_every: Optional[int] = field(default=2, metadata={"help": "save every n generations"})
    cefr_in_system: Optional[int] = field(default=0, metadata={"help": "cefr details in the system prompt (-1: none, 0: descr, k: descr + k examples)"})
    cefr_in_prompt: Optional[int] = field(default=1, metadata={"help": "cefr details in the user prompt (-1: none, 0: descr, k: descr + k examples)"})
    model: Optional[str] = field(default="gpt-4", metadata={"help": "model to use for generation"})




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
    # return ""
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


def get_gpt_prompts(prompt, target_cefr, cefr_in_system=0, cefr_in_prompt=1):
    if pd.isnull(target_cefr):
        cefr_in_system = -1
        cefr_in_prompt = -2
    # else:
        # cefr_in_system = cefr_in_prompt
        # cefr_in_prompt = 1

    prompter = CEFRControlPrompt(cefr_in_system=cefr_in_system, cefr_in_prompt=cefr_in_prompt, max_example_len=1000)
    sysprompt = prompter.get_system_prompt(SYSTEM_PROMPT)
    user_prompt = prompter.get_user_prompt(USER_PROMPT.format(prompt = prompt), level=target_cefr)
    # sysprompt = SYSTEM_PROMPT
    # user_prompt = USER_PROMPT_TEMPLATE.format(CEFR=target_cefr)
    return sysprompt, user_prompt

def generate_at_level(prompt, target_cefr, model, cefr_in_system=0, cefr_in_prompt=1):
    system_prompt, user_prompt = get_gpt_prompts(prompt, target_cefr, cefr_in_system=cefr_in_system, cefr_in_prompt=cefr_in_prompt)
    combined = f"====== System ======\n{system_prompt}\n====== User ======\n{user_prompt}"
    # combined = "" # see above rows
    try:
        # print('*'*100)
        # print(system_prompt)
        # print('~~~~~~~~~~~~~~~')
        # print(user_prompt)
        # print('===================')
        gen = call_gpt(system_prompt, user_prompt, model)

        # print(target_cefr)

        return gen, combined
    except Exception as e:
        return None, f'<ERROR>:{str(e)}' + '\n\n' + combined

def main():
    args: ScriptArguments = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    model = args.model

    df = pd.read_csv(args.to_gen,  dtype={'prompt': str, 'story': str, 'index': int, 'target_cefr': str, 'generation': str})

    tqdm.pandas()


    for i, row in tqdm(df.iterrows(), total=len(df)):
        if not pd.isnull(row['generation']):
            continue

        gen, prompt = generate_at_level(row['prompt'], row['target_cefr'], model, cefr_in_system=args.cefr_in_system, cefr_in_prompt=args.cefr_in_prompt)
        df.loc[i, 'generation'] = gen
        df.loc[i, 'gpt_request'] = prompt if i < 3 else ""
        df.loc[i, "gen_model"] = model
        if i % args.save_every == 0:
            print('saving...')
            df.to_csv(args.to_gen, index=False)


        # time.sleep(1) # avoid rate limit

    df.to_csv(args.to_gen, index=False)


if __name__ == '__main__':
    main()
