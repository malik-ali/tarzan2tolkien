
from glob import glob
import os
import json
import pandas as pd

from tqdm import tqdm

from src.constants import DATA_DIR
import random

CSV_BREAK_LEN = 500000
TOTAL_STORIES = 2745330
N_WANTED = 100000
ACCEPT_P = N_WANTED / TOTAL_STORIES

def sample_subset():
    data_dir = os.path.join(DATA_DIR, "en", "tinystories", "TinyStories_all_data")
    shard_filenames = list(sorted(glob(os.path.join(data_dir, "*.json"))))
    shard_num = 0
    csv = []
    cnt = 0
    for fn in tqdm(shard_filenames):
        with open(fn, "r") as f:
            data = json.load(f)
            print(len(csv))
            for d in data:
                if 'summary' in d and d['source'] == 'GPT-4':
                    obj = {
                        'story': d['story'],
                        'summary': d['summary'],
                        'source': d['source'],
                        'shard': shard_num,
                    }
                    cnt += 1
                    if random.random() < ACCEPT_P:
                        csv.append(obj)

    df = pd.DataFrame.from_records(csv)
    df.to_csv(os.path.join(DATA_DIR, "en", "tinystories", f"data.csv"), index=False)


def main():
    sample_subset()


if __name__ == '__main__':
    main()