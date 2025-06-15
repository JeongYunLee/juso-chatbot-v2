import os
from dotenv import load_dotenv
from langchain_teddynote import logging

# API 키 정보 로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("Agents-Only-Test")

openai_api_key = os.getenv('OPENAI_API_KEY')

##################################################################################
############################ tools: Tavily Searchs  ####################################
##################################################################################
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(k=6)

##################################################################################
############################ tools: Retriever ####################################
##################################################################################

from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma

client = chromadb.PersistentClient('chroma/')
embedding = OpenAIEmbeddings(model='text-embedding-3-large')  
vectorstore = Chroma(client=client, collection_name="49_files_openai_3072", embedding_function=embedding)

retriever = vectorstore.as_retriever(k=3)

from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate

# 문서의 내용을 표시하는 템플릿을 정의합니다.
document_prompt = PromptTemplate.from_template(
    "<document><content>{page_content}</content><source>{source}</source></document>"
)

# retriever 를 도구로 정의합니다.
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    description="use this tool to search for information about Korean Address in the PDF file",
    document_prompt=document_prompt,
)

##################################################################################
############################ tools: Dall-E  ######################################
##################################################################################

from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.tools import tool

# DallE API Wrapper를 생성합니다.
dalle = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="standard", n=1)


# DallE API Wrapper를 도구로 정의합니다.
@tool
def dalle_tool(query):
    """use this tool to generate image from text"""
    return dalle.run(query)

##################################################################################
############################ tools: advanced assistant  ##########################
##################################################################################

from openai import OpenAI

@tool
def advanced_assistant(input, retrieved_data):
    """ 고급 기능(예: 긴 문서 생성, 추론이 필요한 답변 등)을 수행할 수 있는 모델 """
    client = OpenAI()
 
    response = client.chat.completions.create(
        model="o3",
        messages=[
            { "role": "developer", "content": "You are a helpful assistant." },
            {
                "role": "user", 
                "content": input
            }
        ]
    )
    
    result = response.choices[0].message.content
    return result

##################################################################################
############################ tools: file toolkits ##########################
##################################################################################

# from langchain_community.agent_toolkits import FileManagementToolkit

# # 작업 디렉토리 경로 설정
# working_directory = "tmp"

# # 파일 관리 도구 생성(파일 쓰기, 읽기, 디렉토리 목록 조회)
# file_tools = FileManagementToolkit(
#     root_dir=str(working_directory),
#     selected_tools=["write_file", "read_file", "list_directory"],
# ).get_tools()


##################################################################################
############################ tool binders  #######################################
##################################################################################

# tools = file_tools + [
#     retriever_tool,
#     search,
#     dalle_tool,
# ]

tools = [
    retriever_tool,
    search,
    dalle_tool,
    advanced_assistant
]

##################################################################################
############################ Agents  #############################################
##################################################################################

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

# session_id 를 저장할 딕셔너리 생성
store = {}

# 프롬프트 생성
# 프롬프트는 에이전트에게 모델이 수행할 작업을 설명하는 텍스트를 제공합니다. (도구의 이름과 역할을 입력)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant used Korean ONLY. "
            "You are a professional researcher about Korean Address (e.g. 도로명주소, 지번주소, 행정구역, 기타 '주소'와 관련된 산업과 기술 등). You can use the pdf_search tool to search for information about Korean Address in the PDF file. "
            "If you use pdf_search tool, you must state Cites from the reference on the bottom of your answer. "
            "If the information from pdf_search tool is insufficient, you can find further or recent information by using search tool. "
            "If you use search tool, you can add href link. "
            "You can use image generation tool to generate image from text. "
            "You can use advanced_assistant to write long report or reasoning tasks."
            # "You can use file management tool to save your research result into files if the user requests to write a report.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# LLM 생성
llm = ChatOpenAI(model="gpt-4.1")

# Agent 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
)


# session_id 를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # session_id 가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 대화 session_id
    get_session_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)

##############################################################################################################
################################################Chat Interface################################################
##############################################################################################################

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import json
import asyncio
from typing import AsyncGenerator
from langchain_teddynote.messages import AgentStreamParser

# 각 단계별 출력을 위한 파서 생성
agent_stream_parser = AgentStreamParser()


app = FastAPI(title="Juso Chatbot API")

from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ALLOWED_ORIGINS: list = [
        "http://hike.cau.ac.kr",
        "http://localhost:3000",
        "http://localhost:3001"
    ]

@lru_cache()
def get_settings():
    return Settings() 

settings = get_settings()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    score: float
    run_id: str

async def generate_stream(questions: str) -> AsyncGenerator[str, None]:
    try:
        result = agent_with_chat_history.stream(
            {
                "input": questions
            },
            config={"configurable": {"session_id": "abc123"}},
        )
        
        full_response = ""
        for step in result:
            parsed_step = agent_stream_parser.process_agent_steps(step)
            if parsed_step:
                if isinstance(parsed_step, dict):
                    if 'content' in parsed_step:
                        full_response += parsed_step['content']
                    elif 'answer' in parsed_step:
                        full_response += parsed_step['answer']
                    else:
                        full_response += str(parsed_step)
                else:
                    full_response += str(parsed_step)
        
        # 최종 응답을 JSON 형식으로 반환
        response_data = {
            "answer": full_response,
            "run_id": "run_" + str(hash(questions))  # 간단한 run_id 생성
        }
        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
    except Exception as e:
        error_response = {
            "answer": f"Error: {str(e)}",
            "run_id": "error"
        }
        yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"

@app.post("/")
async def stream_responses(request: Request):
    try:
        data = await request.json()
        message = data.get('message')
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # 스트리밍 대신 일반 응답으로 변경
        result = agent_with_chat_history.invoke(
            {
                "input": message
            },
            config={"configurable": {"session_id": "abc123"}},
        )
        
        # 응답 처리
        if isinstance(result, dict) and 'output' in result:
            response_data = {
                "answer": result['output'],
                "run_id": f"run_{hash(message)}"
            }
        else:
            response_data = {
                "answer": str(result),
                "run_id": f"run_{hash(message)}"
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reset")
async def reset_store():
    global store
    store = {}
    return {"status": "Store reset successfully"}

# @app.get("/files")
# async def list_files():
#     try:
#         files = os.listdir(working_directory)
#         return {"files": files}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.get("/files/{filename}")
# async def download_file(filename: str):
#     try:
#         file_path = os.path.join(working_directory, filename)
#         if not os.path.exists(file_path):
#             raise HTTPException(status_code=404, detail="File not found")
#         return FileResponse(file_path, filename=filename)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_agents:app", host="0.0.0.0", port=8000, reload=True)