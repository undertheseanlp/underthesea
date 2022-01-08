import os
import signal
import subprocess
from os.path import dirname, join
from time import sleep
import requests
import yaml
import uuid

CWD = dirname(__file__)


def test_nlu():
    payload = {'text': 'hey there'}
    headers = {'content-type': 'application/json'}
    r = requests.post('http://localhost:5005/model/parse', json=payload, headers=headers)
    print(r.json())


class RasaServer:
    def __init__(self):
        self.p = None

    def start(self):
        self.p = subprocess.Popen(
            'rasa run --enable-api',
            stdout=subprocess.PIPE,
            shell=True,
            cwd=CWD,
            preexec_fn=os.setsid
        )

    def stop(self):
        os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)


class ChatUser:
    def __init__(self, id="test"):
        self.id = id

    def send(self, message):
        payload = {'sender': self.id, 'message': 'hey there'}
        headers = {'content-type': 'application/json'}
        r = requests.post('http://localhost:5005/webhooks/rest/webhook', json=payload, headers=headers)
        if r is None:
            return None
        return r.json()


if __name__ == '__main__':
    # read test story
    rasa_server = RasaServer()
    rasa_server.start()
    wait_seconds = 20
    print(f'Wait {wait_seconds} seconds for server start...')
    sleep(wait_seconds)

    with open(join(CWD, "tests/test_stories.yml"), "r") as f:
        data = yaml.safe_load(f)
    for story in data["stories"]:
        sender_id = str(uuid.uuid4())
        chat_user = ChatUser(sender_id)
        story_name = story['story']
        print(story_name)
        print()
        print('```')
        for step in story['steps']:
            if 'user' in step:
                user_message = step['user'].strip()
                print(f'🗣️: {user_message}')
                r = chat_user.send(user_message)
                if r is None:
                    print('🤖: ')
                else:
                    for bot_response in r:
                        bot_response_message = bot_response['text']
                        print(f'🤖️: {bot_response_message}')
        print('```\n')
    rasa_server.stop()
