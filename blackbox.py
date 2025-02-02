from __future__ import annotations
from typing import List
import asyncio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
import aiohttp
import time
from enum import Enum
import requests
import json

app = FastAPI(title="LLM Proxy Server")

class MessageType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageUrl | None = None

class Message(BaseModel):
    content: str | List[ContentItem] | None = None
    role: MessageType | None = None

    @field_validator("role", mode="before")
    def map_role(cls, v):
        if v == "user":
            return MessageType.USER
        elif v == "assistant":
            return MessageType.ASSISTANT
        elif v == "system":
            return MessageType.SYSTEM
        return v

class ResponseMessage(BaseModel):
    content: str | None = None
    role: MessageType
    tool_calls: List | None = None
    function_call: dict | None = None

    @field_validator("role", mode="before")
    def map_role(cls, v):
        if v == "user":
            return MessageType.USER
        elif v == "assistant":
            return MessageType.ASSISTANT
        return v

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float | None = None
    max_tokens: str | None = "8092"
    
class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

from typing import Union

class Choices(BaseModel):
    message: Union[Message, ResponseMessage]
    finish_reason: str | None = None
    index: int | None = None

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

class BlackboxAI:
    api_endpoint = "https://api.blackbox.ai/api/chat"
    working = True
    supports_stream = False
    supports_system_message = True
    supports_message_history = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: List[Message],
        proxy: str = None,
        **kwargs
    ):
        headers = {
            'Content-Type': 'application/json'
        }
        
        messages_payload = [
            {"content": msg.content, "role": msg.role.value} for msg in messages
        ]
        
        data = {
            "model": model,
            "messages": messages_payload,
            "max_tokens": kwargs.get("max_tokens", "8092")
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(cls.api_endpoint, json=data) as response:
                if response.status == 200:
                    result = await response.text()
                    yield result.strip()
                else:
                    raise Exception(f"Unexpected response status: {response.status}")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    #print(request)
    try:
        # Convert ContentItem to Message
        converted_messages = []
        for msg in request.messages:
            if isinstance(msg.content, list):
                # Combine all ContentItem text fields into a single string
                combined_content = " ".join(
                    content_item.text for content_item in msg.content if content_item.text
                )
                converted_messages.append(Message(
                    content=combined_content,
                    role=msg.role or MessageType.USER
                ))
            else:
                converted_messages.append(msg)
        
        print(converted_messages)
        # Create async generator
        async_gen = BlackboxAI.create_async_generator(
            model=request.model,
            messages=converted_messages
        )
        
        # Get the first (and only) response
        response = None
        async for result in async_gen:
            response = result
            break
            
        if response is None:
            raise HTTPException(status_code=500, detail="No response generated")

        print("response:", response)
            
        chat_response = ChatResponse(
            id=f"chat-{int(time.time())}",
            choices=[
                Choices(
                    message=ResponseMessage(
                        content=response,
                        role=MessageType.ASSISTANT
                    ),
                    finish_reason="stop",
                    index=0
                )
            ],
            created=int(time.time()),
            model=request.model,
            output=response
        )
        
        return chat_response

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Configuration and startup
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)