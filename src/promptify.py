
from src.cefr import CEFR_DESCRIPTIONS, CEFR_LEVELS, CEFR_EXAMPLE_IDXS, load_cefr_examples
# A class to generate prompts and examples for GPT

CEFR_GEN_SYSPROMPT = """
You are a large language model that can generate content at a certain proficiency level suitable for English language learners.
Your goal is to output content and text at the proficiency level specified in the prompt.
"""

class CEFRControlPrompt():
    def __init__(
            self,
            cefr_in_system: int = -1, # -1: no mention, 0: description, k: description + k examples
            cefr_in_prompt: int = -1, # -2: no mention, -1: ask, 0: description, k: description + k examples,
            max_example_len: int = 1000
    ) -> None:
        # TODO by subclass
        if not -1 <= cefr_in_system <= 5:
            raise ValueError("cefr_in_system must be between -1 and 5 (inclusive)")
        if not -2 <= cefr_in_prompt <= 5:
            raise ValueError("cefr_in_prompt must be between -2 and 5 (inclusive)")

        self.cefr_in_system = cefr_in_system
        self.cefr_in_prompt = cefr_in_prompt
        self.cefr_examples = load_cefr_examples(max_example_len)

    def _gen_cefr_info(self, lev, num_examples):
        base = f'## {lev} {CEFR_DESCRIPTIONS[lev]}\n'
        examples = [f'Example {i+1}: {self.cefr_examples[lev][CEFR_EXAMPLE_IDXS[lev][i]]}' for i in range(num_examples)]
        return base + '\n\n'.join(examples) + ('\n' if examples else '')


    def _gen_all_cefr_info(self):
        if self.cefr_in_system == -1:
            return ''

        num_examples = self.cefr_in_system
        cefr = '\n'.join([self._gen_cefr_info(lev, num_examples) for lev in CEFR_LEVELS])

        return  f"""
{CEFR_GEN_SYSPROMPT}
The descriptions of the proficiency levels are given as follows:

{cefr}
--------------------------------------------------
"""


    def get_system_prompt(self, system_prompt) -> str:
        # TODO by subclass
        # system_prompt = self.system_prompt
        if self.cefr_in_system == -1:
            return system_prompt
        all_cefr_info = self._gen_all_cefr_info()

        return all_cefr_info + system_prompt

    def _user_prompt_base(self, level):
        if self.cefr_in_prompt == -1:
            return ''
        else:
            num_examples = self.cefr_in_prompt
            return f"""
As a reminder, {level} proficiency is described as:

{self._gen_cefr_info(level, num_examples)}
---------------------------------------------------
Prompt:
"""

    def get_user_prompt(self: str, user_prompt, level: str) -> str:
        gen_message = f"Generate according to the prompt below but make sure that the generated text is at the {level} level of English proficiency."
        if self.cefr_in_prompt == -2:
            return user_prompt

        if self.cefr_in_prompt == -1:
            return gen_message + '\n' + user_prompt

        base = self._user_prompt_base(level)
        return gen_message + f'\n{base}\n' + user_prompt


if __name__ == "__main__":
    # test
    system_prompt = "This is a system prompt"
    user_prompt = "This is a user prompt"
    cefr_in_system = 0
    cefr_in_prompt = 1
    cp = CEFRControlPrompt(cefr_in_system, cefr_in_prompt, 1000)

    print('='*10  + ' System Prompt ' + '='*10)
    print(cp.get_system_prompt(system_prompt ))
    print()

    for level in CEFR_LEVELS:
        print('='*10  + ' User Prompt ' + level + ' ' + '='*10)
        print(cp.get_user_prompt(user_prompt, level))
        print()
        # break
