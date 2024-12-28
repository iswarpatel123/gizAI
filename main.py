from __future__ import annotations

import asyncio
import json
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from aiohttp import ClientSession
from enum import Enum

# Type definitions
Messages = List[Dict[str, str]]
AsyncResult = asyncio.Future

class MessageType(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    ASSISTANT = "assistant"  # Add assistant

class Message(BaseModel):
    content: str | None = None
    role: MessageType | None = None
    tool_calls: List | None = None
    function_call: dict | None = None

    @field_validator("role", mode="before")
    def map_role(cls, v):
        if v == "user":
            return MessageType.HUMAN
        elif v == "assistant":
            return MessageType.AI
        return v

class ResponseMessage(BaseModel):
    content: str | None = None
    role: MessageType
    tool_calls: List | None = None
    function_call: dict | None = None

    @field_validator("role", mode="before")
    def map_role(cls, v):
        if v == "user":
            return MessageType.HUMAN
        elif v == "assistant":
            return MessageType.ASSISTANT
        elif v == "system":
            return MessageType.SYSTEM
        return v

class Choices(BaseModel):
    finish_reason: str | None = None
    index: int | None = None
    message: ResponseMessage | None = None  # Use ResponseMessage

class Usage(BaseModel):
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_details: List | None = None
    prompt_tokens_details: List | None = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float | None = None

class ChatResponse(BaseModel):
    id: str | None = None
    choices: List[Choices] | None = None
    created: int | None = None
    model: str | None = None
    object: str | None = None
    service_tier: str | None = None
    system_fingerprint: str | None = None
    usage: Usage | None = None
    output: str | None = None

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
            {"type": msg.role.value, "content": msg.content}
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
            print("No response generated")
            raise HTTPException(status_code=500, detail="No response generated")
            
        chat_response = ChatResponse(
            id='chatcmpl-dc4f6c13-7739-4acc-8940-ec822ccb24dc',
            choices=[
                Choices(
                    finish_reason='stop',
                    index=0,
                    message=ResponseMessage(content=response, role=MessageType.ASSISTANT)  # Use ResponseMessage and MessageType.ASSISTANT
                )
            ],
            created=1735359843,
            model=request.model,
            object='chat.completion',
            system_fingerprint=None,
            usage=Usage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
        )
        
        return chat_response

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Configuration and startup
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
