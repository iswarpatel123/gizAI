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
  "model": "deepseek-ai/DeepSeek-V3",
  "max_tokens": "1024"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)