import openai
import os
from os.path import dirname, join
from prompts import Prompt, GenerateStoriesPrompt, GenerateTraingExamplesPrompt, GenerateReponsePrompt, GenerateContinueStoriesPrompt

if __name__ == '__main__':
    with open(join(dirname(__file__), "tmp", "tokens.txt")) as f:
        tokens = f.read()
    openai.api_key = tokens

    log_folder = join(dirname(__file__), "tmp", "logs")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    ##############################
    # Generate Stories
    ##############################
    prompt = GenerateStoriesPrompt(
        log_file=join(log_folder, "generate_dataset_chatgpt.log"),
        openai=openai)
    user_prompt = "Generate rasa chatbot dataset with 10 stories (senarios), each story at leat 7 turns between a Vietnamese chatbot and a user about greeting"
    prompt.generate(user_prompt)

#     prompt = GenerateContinueStoriesPrompt(
#         log_file=join(log_folder, "generate_dataset_chatgpt.log"),
#         openai=openai)
#     user_prompt = """\
# You're building a chatbot about topic love.

# Generate more turns for stories to handle when user said "tôi rất buồn"

# file stories.yml
# ```
# - story: Tìm hiểu về tình yêu
#   steps:
#   - intent: greet
#   - action: utter_greet
#   - intent: ask_about_love
#   - action: utter_ask_about_love
#   - intent: share_experience
#   - action: utter_share_experience
# ```
# """
#     prompt.generate(user_prompt)

#     prompt = GenerateTraingExamplesPrompt(
#         log_file=join(log_folder, "generate_dataset_chatgpt.log"),
#         openai=openai)
#     user_prompt = """> Generate 10 response variation for utter in Vietnamese
# - utter_ask_heartbreak_story
# """

#     prompt = GenerateReponsePrompt(
#         log_file=join(log_folder, "generate_dataset_chatgpt.log"),
#         openai=openai)
#     user_prompt = """> Generate 10 response variation for utter in Vietnamese
# - utter_share_experience
# """
#     prompt.generate(user_prompt)
    