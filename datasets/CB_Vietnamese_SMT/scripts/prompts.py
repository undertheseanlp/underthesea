from os.path import join, dirname
import datetime


class Prompt:
    def __init__(self, system_prompt='default', user_prompt_footer='', log_file=None, openai=None):
        self.log_file = log_file
        if system_prompt == 'default':
            self.system_prompt = """\
            You are a experienced conversation design expert and you are helping a company build a Vietnamese chatbot.
            """
        else:
            self.system_prompt = system_prompt
        self.user_prompt_footer = user_prompt_footer
        self.openai = openai

    def generate(self, user_prompt):
        self.user_prompt = self.generate_user_prompt(user_prompt)
        openai = self.openai
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt}
            ],
            max_tokens=3000
        )
        self.log(response)
        return response

    def generate_user_prompt(self, user_prompt):
        return user_prompt + "\n" + self.user_prompt_footer

    def log(self, response):
        with open(join(dirname(__file__), "tmp", "logs", "generate_dataset_chatgpt.log"), "a") as f:
            f.write("=" * 80 + "\n")
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            f.write("\n> System prompt\n" + self.system_prompt)
            f.write("\n> User prompt\n" + self.user_prompt)
            usage = response["usage"]
            total_tokens = usage["total_tokens"]
            completion_tokens = usage["completion_tokens"]
            prompt_tokens = usage["prompt_tokens"]
            f.write(f"\n> Usage: Total {total_tokens} tokens ({completion_tokens} completion, {prompt_tokens} prompt)\n")
            f.write("\n> Results\n")
            result = response["choices"][0]["message"]["content"]
            f.write(result)
            f.write("\n\n")


class GenerateStoriesPrompt(Prompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Example prompt:
        # Generate rasa chatbot dataset with 10 stories (senarios), each story at leat 7 turns between a Vietnamese chatbot and a user about love topic

        self.user_prompt_footer = """\
with 3 contents

file `intents.yml` with format

```
nlu:
- intent: intent_name_1
examples: |
    - câu 1
    - câu 2
- intent: intent_name_2
examples: |
    - câu 3
    - câu 4
```

file `stories.yml`
file `domain.yml`

- chỉ sinh ra dữ liệu, không giải thích gì thêm
"""

class GenerateContinueStoriesPrompt(Prompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Example prompt:
        # Generate rasa chatbot dataset with 10 stories (senarios), each story at leat 7 turns between a Vietnamese chatbot and a user about love topic

        self.user_prompt_footer = """\
file intent.yml
```
file `intents.yml` with format

```
nlu:
- intent: intent_name_1
examples: |
    - câu 1
    - câu 2
- intent: intent_name_2
examples: |
    - câu 3
    - câu 4
```

- chỉ đưa ra dữ liệu, không giải thích gì thêm (no more explanation)
"""
        


class GenerateTraingExamplesPrompt(Prompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Example
        # Generate 50 training Vietnamese examples for intent
        # - intent: talk_about_love
        #   entities:
        #   - love_status: "sad"

        self.user_prompt_footer = """\
Give answer in rasa format 

```
nlu:
- intent: check_balance
  examples: |
  - Tài khoản của tôi đang có bao nhiêu
  - Tôi muốn kiểm tra số dư tài khoản
- intent: talk_about_love
  examples: |
  - tôi đang [buồn]{"entity":"love_status","value":"sad"} vì chuyện tình yêu
```

- chỉ sinh ra dữ liệu, không giải thích gì thêm
- each text match with entity and value
- mỗi câu là một câu nói của người dùng nói chuyện với chatbot
- nếu dưới intent có entity, thì đối với mỗi giá trị của entity (nếu có), sinh ra ít nhất 10 câu khác nhau cho cùng giá trị
- nếu dưới intent không có entity, thì trong câu không mark entity
"""

class GenerateReponsePrompt(Prompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Example
        # > Generate 10 response variation for utter in Vietnamese
        # - utter_ask_heartbreak_story

        self.user_prompt_footer = """\
Give answer in rasa format 

```
utter_ask_heartbreak_story:
- text: "Tâm sự với mình đi, sao bạn lại buồn vậy?"
- text: Mình có thể giúp bạn bằng cách lắng nghe và chia sẻ những khó khăn của bạn. Hãy kể cho mình nghe đi.
```

- chỉ sinh ra dữ liệu, không giải thích gì thêm (no explanation)
- mỗi câu là một câu nói với chatbot để hỏi hoặc trả lời user
"""
