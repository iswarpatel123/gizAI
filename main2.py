from __future__ import annotations

import asyncio
import json
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from aiohttp import ClientSession
from enum import Enum

#curl -X POST "http://localhost:8000/v1/chat/completions" \     -H "Content-Type: application/json" \
 #    -d '{"model": "qwen-coder-32b", "messages": [{"type": "human", "content": "Are you qwen?"}]}'

 # models - qwen-coder-32b, chat-gemini-flash, claude-haiku, claude-sonnet, chat-o1-mini

# Type definitions
Messages = List[Dict[str, str]]
AsyncResult = asyncio.Future

class MessageType(str, Enum):
    HUMAN = "human"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    type: MessageType
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    mode: str = "plan"
    noStream: bool = True

class ChatResponse(BaseModel):
    output: str

# Provider implementation
class GizAI:
    api_endpoint = "https://app.giz.ai/api/data/users/inferenceServer.infer"
    working = True
    supports_stream = False
    supports_system_message = True
    supports_message_history = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'DNT': '1',
            'Origin': 'https://app.giz.ai',
            'Pragma': 'no-cache',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Not?A_Brand";v="99", "Chromium";v="130"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"'
        }

        data = {
            "model": "chat",
            "baseModel": model,
            "input": {
                "messages": messages,
                "mode": "chat"
            },
            "noStream": True
        }
        
        # Print request body
        print("Request to API endpoint:")
        print(json.dumps(data, indent=2))
        
        async with ClientSession(headers=headers) as session:           
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                if response.status == 201:
                    result = await response.json()
                    yield result['output'].strip()
                else:
                    raise Exception(f"Unexpected response status: {response.status}")

# FastAPI application
app = FastAPI(title="LLM Proxy Server")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    try:
        # Convert the request messages to the format expected by GizAI
        messages = [
            {"type": msg.type, "content": msg.content}
            for msg in request.messages
        ]
        
        # Create async generator
        async_gen = GizAI.create_async_generator(
            model=request.model,
            messages=messages
        )
        
        # Get the first (and only) response
        response = None
        async for result in async_gen:
            response = result
            break
            
        if response is None:
            raise HTTPException(status_code=500, detail="No response generated")
            
        return ChatResponse(output=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuration and startup
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)