import pandas as pd
import os
from src.constants import DATA_DIR
import numpy as np


CEFR_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
CEFR_LEVELS_FG = ['A1.0', 'A1.5', 'A2.0', 'A2.5', 'B1.0', 'B1.5', 'B2.0', 'B2.5', 'C1.0', 'C1.5', 'C2.0', 'C2.5']

def load_cefr_examples(max_len=1000):
    ## --- UPDATE --- ##
    # Use a more comprehensive dataset of CEFR leveled texts

    df = pd.read_csv(os.path.join(DATA_DIR, "en", "cefr_leveled_texts.csv"))

    # df of text,label csv
    # trim text column to max_len and concatenate a "..."
    df['text'] = df['text'].str[:max_len] + '...'

    df = df[df["label"].isin(CEFR_LEVELS)]
    df = df.groupby("label")["text"].apply(list).reset_index(name="examples")
    return dict(zip(df["label"], df["examples"]))

def cefr_to_score(cefr):
    """
    Input should be in format [A-C]\d.\d
    """
    idx = CEFR_LEVELS_FG.index(cefr)
    return round(idx / 2, 1)

def score_to_cefr(cefr_score):
    # if cefr_score is torch, turn it to cpu numpy
    import torch

    if isinstance(cefr_score, torch.Tensor):
        cefr_score = cefr_score.cpu().numpy()

    cefr_score = np.clip(cefr_score, 0, 5.5)
    cefr_score_rnd = round(2 * cefr_score) / 2
    level = CEFR_LEVELS[int(cefr_score_rnd)]
    frac = round(cefr_score_rnd - int(cefr_score_rnd), 2)

    return f"{level}.{str(frac)[2:]}"

# CEFR_EXAMPLE_IDXS = {
#     "A1": [3,4,5,6,7],
#     "A2": [1,2,3,4,5],
#     "B1": [1,2,3,4,5],
#     "B2": [0,1,2,3,4],
#     "C1": [3,4,5,6,7],
#     "C2": [1,2,3,4,5],
# }

CEFR_EXAMPLE_IDXS = {
    "A1": [0],
    "A2": [0],
    "B1": [0],
    "B2": [0],
    "C1": [0],
    "C2": [0],
}

CEFR_DESCRIPTIONS = {
    "A1": """(Beginner)
The writing uses familiar names, words and very simple sentences, for example as seen on notices and posters or in catalogues.
- Includes the top most frequent 1,000 commonly spoken words in the language
- Includes many words and phrases that fall under common early language learning topics (e.g. common greeting, travel, dining, shopping, etc)
- Includes all proper nouns (country names, person names, etc)
- Includes all cognates shared with English
- Includes all words that look similar to English words that share a similar meaning
""",
    "A2": """(Elementary)
The writing involves short, simple texts with specific, predictable information. Examples include simple everyday material such as advertisements, prospectuses, menus and timetables or short simple personal letters.
- Includes the top most frequent 1,000-2,000 commonly spoken words in the language
""",

    "B1": """(Intermediate)
Texts that consist mainly of high frequency everyday or job-related language. These involve descriptions of events, feelings and wishes in personal letters.
- Includes the top 2,000-5,000 commonly spoken words in the language
- Includes several rarer verb tenses (e.g. conditional, subjunctive, etc)
- Includes some relatively common idiomatic phrases
""",
    "B2": """(Upper Intermediate)
Writing as seen in articles and reports concerned with contemporary problems in which the writers adopt particular attitudes or viewpoints. Also includes contemporary literary prose.
- Includes the top 5,000-10,0000 commonly spoken words in the language
""",
    "C1": """(Proficient)
Writing can include long and complex factual and literary texts, with distinctions of style. Examples include specialised articles and longer technical instructions, even when they do not relate to a well-known field.
- Includes the top 10,0000-20,0000 commonly spoken words in the language
""",
    "C2": """(Advanced Proficient)
Includes virtually all forms of the written language, including abstract, structurally or linguistically complex texts such as manuals, specialised articles and literary works.
- Includes esoteric technical language
"""

}



