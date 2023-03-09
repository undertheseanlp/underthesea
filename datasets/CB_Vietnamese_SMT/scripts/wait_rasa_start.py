import time
import requests

if __name__ == '__main__':
    print("Wait for rasa server to start...")
    while True:
        # check if rasa server is running using requests
        # if yes, break
        # if no, sleep 1s
        try:
            r = requests.get('http://localhost:5005')
            if r.status_code == 200:
                print("Rasa server is up and running.")
                break
        except:
            pass
        time.sleep(1)