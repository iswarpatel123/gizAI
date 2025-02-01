from gradio_client import Client

client = Client("Qwen/Qwen2.5-Max-Demo")
result = client.predict(
		query="Hello!!",
		history=[],
		system="You are a helpful assistant.",
		api_name="/model_chat"
)
print(result)