```mermaid
graph
   accTitle: My title here
   accDescr: My description here
   greet --> utter_greet
   utter_greet --> goodbye
   goodbye --> utter_goodbye
   utter_goodbye --> hello
   hello --> utter_hello
   utter_hello --> greet_1
   greet_1 --> utter_greet_1
   utter_greet_1 --> goodbye_1
   goodbye_1 --> utter_goodbye_1
   utter_goodbye_1 --> hello_2
   hello_2 --> utter_hello_2
```