from typing import Optional
import networkx
import yaml


class ChatbotEngine:
    def render_graph(self):
        print(self.history)


class Story:
    def __init__(self, id, data):
        self.id = id
        try:
            self.data = yaml.safe_load(data.strip())[0]
            self.name = self.data['story']
            self.steps = self.data['steps']
            self.paths = []
            for step in self.steps:
                if 'intent' in step:
                    self.paths.append(step['intent'])
                if 'action' in step:
                    self.paths.append(step['action'])
        except Exception as e:
            raise Exception("Cannot parse the data " + str(e))

    def search(self, history) -> Optional[str]:
        try:
            for i in range(len(self.paths)):
                if self.paths[i:i + len(history)] == history:
                    return self.paths[i + len(history)]
            return None
        except:
            return None


class SimpleChatbotEngine(ChatbotEngine):
    def __init__(self):
        super().__init__()
        self.intents = {}
        self.entities = {}
        self.stories = []
        self.history = []

    def random_id(self):
        # generate new_id with 10 digits or lower letters with random
        import string
        import random
        chars = string.ascii_lowercase + string.digits
        length = 10
        id = ''.join(random.choices(chars, k=length))
        return id

    def add_message(self, message):
        pass

    def add_story(self, story_data):
        story_id = self.random_id()
        story = Story(story_id, story_data)
        self.stories.append(story)
        self.update_state(story)
        return story_id

    def update_state(self, story):
        # write story step into graph_input.md
        content = """\
```mermaid
graph
   accTitle: My title here
   accDescr: My description here
"""
        for i in range(len(story.steps) - 1):
            step = story.paths[i]
            next_step = story.paths[i + 1]
            content += f"   {step} --> {next_step}\n"
        content += "```"
        with open("graph_input.md", "w") as f:
            f.write(content)
        # convert graph_input.md to graph_output.svg
        import os
        os.system("mmdc -i graph_input.md -o graph.svg")

    def predict_action(self, history) -> Optional[str]:
        for story in self.stories:
            action = story.search(history)
            if action:
                return action
        return None

    def get_response(self, message):
        text = "hello"
        self.history.append(message)
        self.history.append(text)
        response = [
            {"text": "text"}
        ]
        return response

# Prompt Interactive Learning


class PILEngine():
    def __init__(self):
        self.history = []
        self.intents = {}
        self.entities = {}
        self.stories = {}

    def get_response(self, message, chatbot_engine=None):
        if message == ":magic":
            text = """\
[PIL]
<br/><br/>
**Magic** from Prompt Interactive Learning 🪄🪄🪄
"""
        elif message.startswith(":new story") or message.startswith(":n story"):
            data = "\n".join(message.strip().split("\n")[1:])
            story_id = chatbot_engine.add_story(data)
            text = f"🪄 New story **{story_id}** is added."
        else:
            text = "❌ Unknown command"
        response = [
            {"text": text}
        ]

        return response

    def get_display_input(self, command):
        lines = command.split("\n")
        lines[0] = "**" + lines[0] + "**"
        return "<br/>".join(lines)
