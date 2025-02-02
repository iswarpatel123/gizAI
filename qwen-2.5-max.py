from __future__ import annotations
from typing import List
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import time
from enum import Enum
from gradio_client import Client

app = FastAPI(title="Qwen 2.5 Max")

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
    tool_calls: List | None = None
    function_call: dict | None = None

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

class QwenAI:
    api_endpoint = "Qwen/Qwen2.5-Max-Demo"
    working = True
    supports_stream = False
    supports_system_message = True
    supports_message_history = True

    @classmethod
    def convert_messages_to_history(cls, messages: List[Message]):
        history = []
        system_prompt = "You are a helpful assistant."
        
        # Extract system message if present
        for msg in messages:
            if msg.role == MessageType.SYSTEM:
                system_prompt = msg.content
                break
        
        # Convert chat messages to history format
        for i in range(len(messages)-1):
            if messages[i].role == MessageType.USER and messages[i+1].role == MessageType.ASSISTANT:
                history.append([messages[i].content, messages[i+1].content])
        
        # Return the last user message separately if it exists
        last_user_msg = ""
        if messages and messages[-1].role == MessageType.USER:
            last_user_msg = messages[-1].content
            
        return history, last_user_msg, system_prompt

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: List[Message],
        **kwargs
    ):
        client = Client(cls.api_endpoint)
        history, query, system = cls.convert_messages_to_history(messages)
        
        try:
            result = client.predict(
                query=query if query else "Hello!",
                history=history,
                system=system,
                api_name="/model_chat"
            )
            
            # Result is a tuple of (input_text, chat_history, system_prompt)
            # We want the response from the last interaction
            if isinstance(result[1], list) and len(result[1]) > 0:
                response = result[1][-1][1] if result[1][-1][1] else ""
            else:
                response = ""
                
            yield response

        except Exception as e:
            raise Exception(f"Error calling Qwen API: {str(e)}")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    print(request)
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
        
        async_gen = QwenAI.create_async_generator(
            model=request.model,
            messages=converted_messages
        )
        
        response = None
        async for result in async_gen:
            response = result
            break
            
        if response is None:
            raise HTTPException(status_code=500, detail="No response generated")
            
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
        raise HTTPException(status_code=500, detail=str(e))

# Configuration and startup
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
