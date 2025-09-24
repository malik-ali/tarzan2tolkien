
# From Tarzan to Tolkien: Controlling the Language Proficiency Level of LLMs for Content Generation

[Ali Malik](https://malikaliraza.com), [Stephen Mayhew](https://mayhewsw.github.io), [Chris Piech](https://stanford.edu/~cpiech/bio/index.html), and [Klinton Bicknell](https://www.klintonbicknell.com).

Work done at Duolingo. Presented in Findings of the Association for Computational Linguistics: ACL 2024

- **Paper**: [https://aclanthology.org/2024.findings-acl.926/](https://aclanthology.org/2024.findings-acl.926)
- **Arxiv**: [https://arxiv.org/abs/2406.03030](https://arxiv.org/abs/2406.03030)


Our main CEFR Aligned Language Model (CaLM) can be found on HuggingFace at: [https://huggingface.co/malikali/CEFR-Aligned-LM](https://huggingface.co/malikali/CEFR-Aligned-LM)

## Overview of Code
This repository contains code for the paper "From Tarzan to Tolkien: Controlling the Language Proficiency Level of LLMs for Content Generation". The code is organized into several directories:
- `scripts/`: Contains all scripts for training, generating, and evaluating models.
- `src/`: Contains the main source code for the scripts.
- `data/`: Folder to store the datasets used in the experiments.
- `README.md`: This file, providing an overview of the repository and its contents.

## Installation
This codebase uses conda environments and installs the directory as a local python package. To set up the environment, run the following commands from the root of the repository:

```bash
conda create -n tarzan2tolkien python=3.10
conda activate tarzan2tolkien
pip install -r requirements.txt
pip install -e .

# "Download nlp packages"
python -m spacy download en_core_web_sm
python -m nltk.downloader popular

```

Every time you open the project, you should activate the environment with:

```bash
conda activate tarzan2tolkien
```

You will also have to login to wandb and huggingface + configure accelerate.

```bash
wandb login
huggingface-cli login
accelerate config
```


## Setup and Data
You should look at `src/constants.py` to set the `ROOT_DIR` variable to point to the root of your local copy of the repo and adjust any other paths as necessary. Some parts of the codebase may require additional data files that we are unable to share but should be easy to find alternatives online. This sections are marked with a `## --- UPDATE --- ##` comment.

We make use of the TinyStories dataset from [https://arxiv.org/pdf/2305.07759](https://arxiv.org/pdf/2305.07759). You can download and prepare the data as follows:

```bash
mkdir -p data/en/tinystories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz -O data/en/tinystories/TinyStories_all_data.tar.gz
mkdir -p data/en/tinystories/TinyStories_all_data
tar -xvzf data/en/tinystories/TinyStories_all_data.tar.gz -C data/en/tinystories/TinyStories_all_data
rm data/en/tinystories/TinyStories_all_data.tar.gz
python src/tinystories/combine_data.py
```


## Training a CEFR Scorer

We train a CEFR scoring function using Ridge Regression with simple text features. The crux of the featurisation logic can be found in `src/cefr_scorer/featurise.py`. To train the model, you will have to collect a dataset of CEFR leveled texts. One really amazing recent resource is [UniversalCEFR](https://huggingface.co/UniversalCEFR).


## Closed Models / Prompting

To generate stories with closed source models like GPT-4, we first generate prompt requests using the following command. This requests file acts as a queue of prompts to generate stories for.

```bash
python scripts/prompting/1_gen_prompt_requests.py \
  --prompt_data data/en/tinystories/data.csv \
  --outfile out/tinystories/model_name/generations.csv \
  --n_per_cefr 1
```

We can then use the following command to generate stories with GPT-4. This script writes back to the csv file with the generated stories so can be interrupted and resumed. Make sure to set the OPENAI_API_KEY environment variable before running this command.

```bash
python scripts/prompting/2_generate.py \
  --to_gen out/tinystories/model_name/generations.csv \
  --cefr_in_system -1 --cefr_in_prompt 0
```

The flags --cefr_in_system and --cefr_in_prompt control how much detail is included in the prompt request. See the `CEFRControlPrompt` class in `src/promptify.py` for more details.

## Opensource models

### Finetuning

This code uses accelerate to handle multi-gpu training. To do supervised fine-tuning of an opensource model, run the following command. Make sure to adjust the parameters according to your needs.

The fintuning data is loaded from `src/opensource_utils.py` and just points to a csv with columns: 'prompt', 'generation', 'target_cefr', and 'split'.

```bash
accelerate launch scripts/opensource/supervised_finetune.py \
  --model_name=meta-llama/Llama-2-7b-hf \
  --wandb_name=tinystories-llama2-7b-ft \
  --output_dir models/tinystories/opensource/finetuned/llama2_7b \
  --learning_rate=1e-4 \
  --weight_decay=5e-1 \
  --batch_size=2 \
  --eval_freq=40 \
  --save_freq=40
```

After training you can merge the peft adapter with the base model and upload to huggingface model hub:

```bash
python scripts/opensource/merge_peft_adapter.py \
  --checkpoint_dir models/tinystories/opensource/finetuned/llama2_7b/checkpoint-1200 \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --output_name=hfaccount/llama2-cefrstories-ft
```

### RLHF with Scoring Function

To run RLHF with a CEFR scoring function, you can use the following command. The scorer is hardcoded in the script but you can swap it out with any other scoring function you want.

```bash
accelerate launch scripts/opensource/rlhf.py \
    --model_name=hfaccount/llama2-cefrstories-ft \
    --output_dir=models/tinystories/opensource/rlhf/llama2-cefrstories-rlhf \
    --batch_size=32 \
    --output_max_length=256
```


### Generation

To generate stories with a trained model, we first generate prompt requests using the following command:

```bash
python scripts/prompting/1_gen_prompt_requests.py \
  --prompt_data data/en/tinystories/data.csv \
  --outfile out/tinystories/opensource/model_name/eval.csv \
  --n_per_cefr 1
```

This requests file acts as a queue of prompts to generate stories for. We can the use the following command to generate stories with a trained model. This script writes back to the csv file with the generated stories so can be interrupted and resumed.

```bash
python scripts/opensource/2_generate.py \
  --to_gen out/tinystories/opensource/model_name/eval.csv \
  --model_dir hfaccount/llama2-cefrstories-rlhf \
  --save_every 100 \
  --is_local_peft False
```

The next command extracts the story from the model output.

```bash
python scripts/opensource/3_extract_gen_stories.py --gen_file ./out/tinystories/opensource/model_name/eval.csv
```


## Scoring

We can use the following command to score the generated stories with the CEFR scorer.


### CEFR Scorer

```bash
python scripts/cefr_scorer/eval.py \
  --input_file ./out/tinystories/opensource/model_name/eval.csv \
  --output_file ./out/tinystories/opensource/model_name/scored.csv \
  --columns generation
```

### Semantic Similarity Scorer

Make sure to export the OPENAI_API_KEY environment variable before running this command.

```bash
python scripts/semantic_scorer/gpt_score.py \
  --input_file ./out/tinystories/opensource/model_name/scored.csv \
  --output_file ./out/tinystories/opensource/model_name/gpt_sem_scored.csv \
  --gen_column generation \
  --prompt_column prompt
```

