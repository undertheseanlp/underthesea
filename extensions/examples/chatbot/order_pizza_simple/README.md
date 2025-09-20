# Order Pizza Vietnamese Chatbot Example

### Usage

``` 
cd underthesea/examples/chatbot/order_pizza_simple
rasa train
rasa x 
```

### Examples

Script 1

```
🗣️: tôi muốn mua pizza
🤖: Bạn muốn mua loại pizza nào?
🗣️: hải sản
🤖: Bạn chọn cỡ gì?
🗣️: lớn
🤖: Bạn vừa đặt một pizza hải sản cỡ lớn.
```

Script 2

```
🗣️: tôi muốn mua pizza hải sản
🤖: Bạn chọn cỡ gì?
🗣️: to
🤖: Bạn vừa đặt một pizza hải sản cỡ to.
```

Script 3

``` 
🗣️: tôi muốn mua pizza hải sản to
🤖: Bạn vừa đặt một pizza hải sản cỡ to.
```

Script 4: Validating Form Input

```
🗣️: tôi muốn mua pizza
🤖: Bạn muốn mua loại pizza nào?
🗣️: Bò xốt tiêu đen
🤖: Nhà hàng hiện tại có 3 loại: hải sản, phô mai, gà nướng
🤖: Bạn muốn mua loại pizza nào?
🗣️: Gà nướng
🤖: Bạn chọn cỡ gì?
🗣️: vừa
🤖: Bạn vừa đặt một pizza Gà nướng cỡ vừa.
```