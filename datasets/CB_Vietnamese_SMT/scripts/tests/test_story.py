import unittest
from ..chat_engine import Story

class TestStory(unittest.TestCase):
    def test_init(self):
        data = """\
- story: Greet and say goodbye
  steps:
  - intent: greet
  - action: utter_greet
  - intent: goodbye
  - action: utter_goodbye
"""
        id = "123"
        story = Story(id, data)
        self.assertEqual(4, len(story.data[0]['steps']))
    
    def test_search(self):
        data = """\
- story: Greet and say goodbye
  steps:
  - intent: greet
  - action: utter_greet
  - intent: goodbye
  - action: utter_goodbye
"""
        id = "123"
        story = Story(id, data)
        history = ["greet"]
        action = story.search(history)
        self.assertEqual("utter_greet", action)