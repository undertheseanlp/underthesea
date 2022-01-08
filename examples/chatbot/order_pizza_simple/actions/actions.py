# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
from typing import Text, Dict, Any

from rasa_sdk import FormValidationAction, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ValidationSimplePizzaForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_simple_pizza_form"

    def validate_pizza_s1_type(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict
    ) -> Dict[Text, Any]:
        """Validate `pizza_s1_type` value."""
        ALLOWED_PIZZA_TYPES = ["hải sản", "phô mai", "gà nướng"]
        if slot_value.lower() not in ALLOWED_PIZZA_TYPES:
            n = len(ALLOWED_PIZZA_TYPES)
            text = f"Nhà hàng hiện tại có {n} loại: " + ", ".join(ALLOWED_PIZZA_TYPES).lower()
            dispatcher.utter_message(text)
            return {"pizza_s1_type": None}
        return {"pizza_s1_type": slot_value}
