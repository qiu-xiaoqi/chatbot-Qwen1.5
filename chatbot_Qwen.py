import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModel
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import os
from fastapi.responses import HTMLResponse


"""
flowchart TD
    A[启动应用] --> B[添加CORS中间件]
    B --> C[读取HTML文件]
    C --> D[定义GET请求处理]
    D --> E[定义WebSocket端点]
    E --> F{接收连接}
    F -->|成功| G[接受连接]
    G --> H{接收JSON消息}
    H -->|有消息| I[解析消息]
    I --> J[调用模型生成响应]
    J --> K{遍历响应}
    K -->|是| L[发送部分响应]
    L --> M[继续遍历]
    M --> K
    K -->|否| N[发送结束状态]
    N --> O[等待下一条消息]
    O --> H
    H ---->|无消息| P[断开连接]
    B --> Q[定义HTTP接口]
    Q --> R[定义/v1/models]
    R --> S[返回模型列表]
    Q --> T[定义/v1/chat/completions]
    T --> U{解析请求}
    U -->|有效| V[调用模型生成响应]
    V --> W{是否流式响应}
    W -->|是| X[创建流式响应]
    X --> Y[发送流式响应]
    W -->|否| Z[创建非流式响应]
    Z --> AA[发送非流式响应]
"""

# lifespan：异步上下文管理器函数，用于管理 FastAPI 应用程序的生命周期。
# 在 yield 之前的代码会在应用程序启动时执行。
# 在 yield 之后的代码会在应用程序关闭时执行。
@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

# 中间件，跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许跨域访问的域名，* 表示允许所有域名访问（不推荐在生产环境中使用，会降低安全性）
    allow_credentials=True,  # 允许携带认证信息（cookies、HTTP 认证和bearer token）
    allow_methods=["*"],  # 允许的 HTTP 方法
    allow_headers=["*"],  # 允许的 HTTP 请求头
)

# 可以自动获得数据验证、序列化和反序列化的功能。
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


# 用户、助手或系统的聊天消息
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


# 表示增量消息，可选包含角色和内容
class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


# 表示聊天完成请求，包含模型、消息列表和其它参数
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


# 表示非流式响应的选择
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


# 表示流式响应的选择
class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


# 表示聊天完成响应，包含模型、对象类型、选择列表和创建时间
class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i + 1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i + 1].content])

    if request.stream:
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(query: str, history: List[List[str]], model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'


with open('websocket_demo.html') as f:
    html = f.read()


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    input: JSON String of {"query": "", "history": []}
    output: JSON String of {"response": "", "history": [], "status": 200}
        status 200 stand for response ended, else not
    """
    await websocket.accept()
    try:
        while True:
            json_request = await websocket.receive_json()
            query = json_request['query']
            history = json_request['history']
            for response, history in model.stream_chat(tokenizer, query, history=history):
                await websocket.send_json({
                    "response": response,
                    "history": history,
                    "status": 202,
                })
            await websocket.send_json({"status": 200})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    pretrain_path = "Qwen/Qwen1.5-1.8B-Chat"
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"Path {pretrain_path} does not exist")
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    model = AutoModel.from_pretrained(pretrain_path).cuda()
    model = model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)


