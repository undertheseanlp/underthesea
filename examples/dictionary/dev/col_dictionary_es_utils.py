import json
import requests

url = "http://localhost:9200/_all/_settings"
payload = json.dumps({"index.blocks.read_only_allow_delete": "false"})
headers = {
    'Content-Type': "application/json",
}
response = requests.request("PUT", url, data=payload, headers=headers)
print(response.status_code)
print(response.text)
