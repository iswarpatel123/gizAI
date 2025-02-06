from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from giz import ChatResponse, Choices, ResponseMessage, MessageType, ChatRequest
import json
import aiohttp
import asyncio
import re

app = FastAPI()

def extract_json(s):
    """Extracts the first valid JSON object from a string.

    Args:
        s: The string to extract from.

    Returns:
        A tuple (json_string, remaining_string), where:
        - json_string is the extracted JSON object as a string.
        - remaining_string is the rest of the input string after the JSON object.

    Raises:
        ValueError: If no valid JSON object is found.
    """
    match = re.search(r"\{.*\}", s)
    if not match:
        raise ValueError("No JSON object found")

    json_string = match.group(0)
    remaining_string = s[match.end():]
    return json_string, remaining_string

# Configuration for your LLM inference endpoint
LLM_ENDPOINT = "https://chat.sonus.ai/chat.php"  # Replace with your actual LLM endpoint
API_KEY = "your-api-key"  # Replace with your actual API key if authentication is required

# Helper function to map the model if needed
def map_model(model: str) -> str:
    return model  # Adjust mapping if necessary

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    try:
        # Convert the request messages to the format expected by Sonus
        messages = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            elif isinstance(msg.content, list):
                content_parts = []
                for content_item in msg.content:
                    if content_item.type == "text" and content_item.text:
                        content_parts.append(content_item.text)
                    elif content_item.type == "image_url" and content_item.image_url:
                        content_parts.append(content_item.image_url.url)
                messages.append({
                    "role": msg.role.value,
                    "content": "\n".join(content_parts)
                })

        # Prepare the payload for the LLM inference endpoint
        llm_payload = {
            "model": map_model(request.model),
            "messages": messages,
            "reasoning": True,  # Default reasoning to True
        }

        # Headers for the LLM API request
        headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "connection": "keep-alive",
            "content-length": "433",
            "content-type": "application/json",
            "cookie": "logged_in_email=iswarpatel%40yahoo.com; id=xSJDkCto9J; username=iswarpatel; name=Iswar%20P; timezone=America%2FNew_York; location=Ashburn%2C%20United%20States",
            #"host": "chat.sonus.ai", # Removed host, aiohttp adds automatically
            "origin": "https://chat.sonus.ai",
            "referer": "https://chat.sonus.ai/",
            "sec-ch-ua": '"Not A(Brand)";v="8", "Chromium";v="132", "Google Chrome";v="132"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
        }

        # Send the request to the LLM endpoint asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_ENDPOINT, headers=headers, json=llm_payload) as resp:
                if resp.status != 200:
                    content = await resp.text()
                    print(f"Error response from LLM endpoint: {content}")
                    raise HTTPException(status_code=resp.status, detail="Failed to get response from LLM endpoint")

                # Accumulate the streamed response
                llm_response_text = ""
                try:
                    async for chunk in resp.content.iter_any():
                        llm_response_text += chunk.decode('utf-8')
                        #print(f"Received chunk: {chunk.decode('utf-8')}")

                except Exception as e:
                    print(f"Error during stream reading: {e}")
                    raise HTTPException(status_code=500, detail=f"Error during stream reading: {e}")

                #print("complete response", llm_response_text)
                titles = []
                # Extract JSON string from the response text
                processed_text = llm_response_text
                while True:
                    try:
                        # Find the start of the JSON object
                        json_start = processed_text.find('{')
                        if json_start == -1:
                            break  # No more JSON objects found

                        json_string, processed_text = extract_json(processed_text[json_start:])

                        llm_response = json.loads(json_string)


                        # Extract title from the received JSON object
                        title = llm_response.get("title") if isinstance(llm_response, dict) else None

                        if title is not None:
                            titles.append(title)


                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error parsing JSON: {e}")
                        print(f"Received: {llm_response_text}")
                        raise HTTPException(status_code=500, detail="Invalid JSON response from LLM endpoint")
                combined_title = " ".join(titles)
                # Map the LLM's response to the ChatResponse format
                chat_response = ChatResponse(
                    choices=[
                        Choices(
                            finish_reason='stop',
                            index=0,
                            message=ResponseMessage(content=combined_title, role=MessageType.ASSISTANT)
                        )
                    ],
                )

                return chat_response

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")

# If you're running this script directly, use uvicorn to serve the app.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
