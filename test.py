import requests
import json

url = "https://api.blackbox.ai/api/chat"

payload = json.dumps({
  "messages": [
    {
      "content": "You are a helpful assistant.",
      "role": "system"
    },
    {
      "content": "which LLM are you?",
      "role": "user"
    }
  ],
  "model": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
  "max_tokens": "8092"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)