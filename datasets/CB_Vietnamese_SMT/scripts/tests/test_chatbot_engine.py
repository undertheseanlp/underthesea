# create unittest
import unittest
from chat_engine import SimpleChatbotEngine

class ChatbotEngineTest(unittest.TestCase):
    def test(self):
        chatbot = SimpleChatbotEngine()
        self.assertEqual(1, 1)

    def test_add_story(self):
        chatbot = SimpleChatbotEngine()
        story_data = """\
- story: Greet and say goodbye
  steps:
  - intent: greet
  - action: utter_greet
  - intent: goodbye
  - action: utter_goodbye
"""
        chatbot.add_story(story_data=story_data)
        action = chatbot.predict_action(history=[
            "greet"
        ])
        self.assertEqual("utter_greet", action)
    
    def test_update_state(self):
        chatbot = SimpleChatbotEngine()
        story_data = """\
- story: Greet and say goodbye
  steps:
  - intent: greet
  - action: utter_greet
  - intent: goodbye
  - action: utter_goodbye
"""
        chatbot.add_story(story_data=story_data)
        story = chatbot.stories[0]
        chatbot.update_state(story)