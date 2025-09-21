# Vietnamese Greetings Chatbot Example

### Usage

Run agent

``` 
cd underthesea/examples/chatbot/baggage_claim
rasa run actions 
rasa train
rasa x 
```

Run tests

```
cd underthesea/examples/chatbot/baggage_claim
rasa data validate
rasa test core --stories tests
```
### Examples

Script 1

``` 
🗣️: chào bạn
🤖: Chào bạn! Bạn thế nào?
🗣️: tớ vui lắm
🤖: Tuyệt vời!
```

Script 2

``` 
🗣️: hi
🤖: Chào bạn! Bạn thế nào?
🗣️: chán lắm
🤖: Đừng buồn nhé bạn hỡi
🤖: Bạn đã đỡ buồn chưa
🗣️: rồi
🤖: Tuyệt vời!
```

Script 3

``` 
🗣️: hi
🤖: Chào bạn! Bạn thế nào?
🗣️: chán lắm
🤖: Đừng buồn nhé bạn hỡi
🤖: Bạn đã đỡ buồn chưa
🗣️: chưa
🤖: Tạm biệt bạn!
```

Script 4

``` 
🗣️: đây là người hay máy
🤖: Mình là bot nhé.
```

Script 5

``` 
🗣️: tạm biệt nhé
🤖: Tạm biệt bạn!
```
