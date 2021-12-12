# Vietnamese Chatbot

Migrating from [https://github.com/undertheseanlp/chatbot](https://github.com/undertheseanlp/chatbot)

In this example, we will create some simple demo for Vietnamese Chatbot with Rasa

## List Chatbot 

<table>
<thead>
  <tr>
    <th>Chatbot</th>
    <th>Concept</th>
    <th>Statistics</th>
    <th>Scripts</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <a href="top_up">Top Up</a>
    </td>
    <td>Domains, Intents, Slots</td>
    <td>2 intents, 1 slots</td>
    <td></td>
  </tr>
  <tr>
    <td>Order Pizza Simple</td>
    <td>Forms, Validating Form Input, Actions</td>
    <td>1 form, 2 intents, 2 slots</td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>

## Usage

Environment

* Rasa 2.8.2
* Rasa-X 0.42.0
* Python 3.8
* Ubuntu

Setup Environment

```
conda create -n chatbot python=3.8
conda activate chatbot
pip install --upgrade pip==20.2    # more detail https://forum.rasa.com/t/pip-takes-long-time/39274/3
pip install rasa==2.8.2
pip install rasa-x==0.42.0 --extra-index-url https://pypi.rasa.com/simple
pip install sanic-jwt==1.6.0       # more detail https://forum.rasa.com/t/pip-install-rasa-x-not-working-local-mode-still-takes-to-long/48247/5
pip install questionary==1.5.1     # more detail https://forum.rasa.com/t/error-this-event-loop-is-already-running/24017/11
```

Run Rasa X

```
conda activate chatbot 
cd underthesea/examples/chatbot/topup
rasa x
```
