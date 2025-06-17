# 서버 코드 수정 (main.py)

import uuid, os
from dotenv import load_dotenv
from typing import TypedDict
from typing import Annotated

from openai import OpenAI
from langchain_openai import ChatOpenAI
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import Client
from langchain_teddynote import logging
from langchain_core.tracers.context import collect_runs

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig  
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph.message import add_messages
from operator import itemgetter

from langchain.agents import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from langchain_teddynote import logging

# .env 파일 활성화 & API KEY 설정
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
logging.langsmith("hike-jusochatbot-demo")

llm_4o = ChatOpenAI(model="gpt-4o", temperature=0)

class GraphState(TypedDict):
    question: str # 질문
    q_type: str  # 질문의 유형
    context: list | str  # 문서의 검색 결과
    answer: str | list[str]   # llm이 생성한 답변
    relevance: str  # 답변의 문서에 대한 관련성 (groundness check)
    session_id: str  # 세션 ID 추가

# 전역 변수들
store = {}

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
        # print(f"🆕 새로운 세션 히스토리 생성: {session_ids[:8]}...")
    else:
        pass
        # print(f"📚 기존 세션 히스토리 로드: {session_ids[:8]}... (메시지 수: {len(store[session_ids].messages)})")
    return store[session_ids]

# 새로운 세션 ID 생성 함수
def generate_session_id():
    return str(uuid.uuid4())

#######################################################################
############################ nodes: Router ############################
#######################################################################

class Router(BaseModel):
    type: str = Field(description="type of the query that model choose")

router_output_parser = JsonOutputParser(pydantic_object=Router)
format_instructions = router_output_parser.get_format_instructions()

router_prompt = PromptTemplate(
    template="""
            You are an expert who classifies the type of question. There are two query types: ['general', 'domain_specific']

            [general]
            Questions unrelated to addresses, such as translating English to Korean, asking for general knowledge (e.g., "What is the capital of South Korea?"), or queries that can be answered through a web search.

            [domain_specific]
            Questions related to addresses, such as concepts, definitions, address-related data analysis, or reviewing properly written addresses (e.g., "수지구는 자치구이니 일반구이니?", "특별시에 대해서 설명해줘", "주소와 주소정보의 차이점은?").

            <Output format>: Always respond with either "general" or "domain_specific" and nothing else. {format_instructions}
            <chat_history>: {chat_history}
            
            <Question>: {query} 
            """,
    input_variables=["query", "chat_history"],
    partial_variables={"format_instructions": format_instructions},
)

def router(state: GraphState) -> GraphState:
    chain = router_prompt | llm_4o | router_output_parser
    
    router_with_history  = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="chat_history",
    )
    
    router_result = router_with_history.invoke(
        {"query": state["question"]}, 
        {'configurable': {'session_id': state["session_id"]}}
    )
    state["q_type"] = router_result['type']
    return state

def router_conditional_edge(state: GraphState) -> GraphState:
    q_type = state["q_type"].strip()
    return q_type

##################################################################################
############################ nodes: Retrieve Document ############################
##################################################################################

client = chromadb.PersistentClient('chroma/')
embedding = OpenAIEmbeddings(model='text-embedding-3-large')  
vectorstore = Chroma(client=client, collection_name="49_files_openai_3072", embedding_function=embedding)

def retrieve_document(state: GraphState) -> GraphState:
    retrieved_docs_with_score = vectorstore.similarity_search_with_score(state["question"], k=3)

    serialized_docs = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        }
        for doc, score in retrieved_docs_with_score
    ]

    return {**state, "context": serialized_docs}

#########################################################################
############################ nodes: Verifier ############################
#########################################################################

class Verifier(BaseModel):
    type: str = Field(description="verify that retrieved data is sufficient to answer the query")

verifier_output_parser = JsonOutputParser(pydantic_object=Verifier)
format_instructions = verifier_output_parser.get_format_instructions()

verifier_prompt = PromptTemplate(
    template="""
            You are an expert who verity the retrieved data's quality and usefullness to answer the query. There are two query types: ['sufficient', 'insufficient', 'unsuitable']

            [sufficient]
            When the retrieved data is sufficient to answer the query.
            
            [insufficient]
            When the retrieved data is insufficient to answer the query, triggering additional actions or tool usage:
            1.	When the context information is inadequate to respond to the query, requiring further steps (e.g., search).
            2.	When the query involves tasks beyond simple address-related information retrieval, such as report generation or image creation.

            [unsuitable]
            When the retrieved data is not suitable to answer the query.
            
            <Output format>: Always respond with either "sufficient", "insufficient" or "unsuitable" and nothing else. {format_instructions}
            
            <Question>: {query} 
            <Retrieved data>: {retrieved_data}
            """,
    input_variables=["query", "retrieved_data"],
    partial_variables={"format_instructions": format_instructions},
)

def verifier(state: GraphState) -> GraphState:
    chain = verifier_prompt | llm_4o | verifier_output_parser
    verified = chain.invoke(
        {"query": state["question"], "retrieved_data": state["context"]}, 
        {'configurable': {'session_id': state["session_id"]}}
    )
    state["relevance"] = verified['type']
    return state

def verifier_conditional_edge(state: GraphState) -> str:
    verified_result = state["relevance"].strip()
    
    if verified_result not in ["sufficient", "insufficient", "unsuitable"]:
        raise ValueError(f"Unexpected verifier result: {verified_result}")

    return verified_result
 
############################ tools ############################

@tool
def search_on_web(input):
    """ 실시간 정보, 최신 정보 등 웹 검색이 필요한 질문에 답변하기 위해 사용하는 도구 """
    search_tool = TavilySearchResults(max_results=5)
    search_result = search_tool.invoke({"query": input})
    return search_result

from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.tools import tool

dalle = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="standard", n=1)

@tool
def dalle_tool(query):
    """use this tool to generate image from text"""
    return dalle.run(query)

@tool
def advanced_assistant(input, retrieved_data):
    """ 고급 기능(예: 보고서 생성, 긴 문서 생성, 추론이 필요한 답변 등)을 수행하기 위한 모델 """
    client = OpenAI()
 
    response = client.chat.completions.create(
        model="o3",
        messages=[
            { "role": "developer", "content": "You are a helpful assistant." },
            {
                "role": "user", 
                "content": f"query: {input}\n\n retrieved data: {retrieved_data}"
            }
        ]
    )
    
    result = response.choices[0].message.content
    return result

@tool
def image_explainer(query, image_url):
    """ 이미지 설명 생성기. Use after the image_generator tool make the image output. Use the url of the image_generator tool. """
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0]
    
tools = [search_on_web, dalle_tool, advanced_assistant, image_explainer]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant used Korean ONLY. "
            "When `Relevance` is sufficient, you may provide the answer directly. "
            "When `Relevance` is insufficient, you MUST use another tools like search_on_web or advanced_assistant. "
            "If you use `Retrieved Data`, you must state Cites from the reference on the bottom of your answer. "
            "If the information from pdf_search tool is insufficient, you can find further or recent information by using search tool. "
            "If you use search tool, you can add href link. "
            "You can use image generation tool to generate image from text. "
            "You can use advanced_assistant to write long report or reasoning tasks."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("human", "{retrieved_data}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

def agent(state: GraphState) -> GraphState:
    llm = ChatOpenAI(model="gpt-4.1")
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=100,
        max_execution_time=100,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        history_messages_key="chat_history",
    )

    result = agent_with_history.invoke(
        {"input": state["question"], "retrieved_data": state["context"], "relevance": state["relevance"]}, 
        {'configurable': {'session_id': state["session_id"]}}
    )
    state['answer'] = result['output']

    return state

########################################################################
############################ Workflow Graph ############################
########################################################################

workflow = StateGraph(GraphState)

workflow.add_node("Router", router)
workflow.add_node("Retrieved Data", retrieve_document)
workflow.add_node("Agent", agent)
workflow.add_node("Verifier", verifier)

workflow.add_conditional_edges(
    "Router",
    router_conditional_edge,
    {"domain_specific": "Retrieved Data",  "general": "Agent"},
)

workflow.add_edge("Retrieved Data", "Verifier")
workflow.add_edge("Verifier", "Agent")
workflow.add_edge("Agent", END)

workflow.set_entry_point("Router")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)    

##############################################################################################################
################################################Chat Interface################################################
##############################################################################################################

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Juso Chatbot API")

from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://localhost:3000",
        "https://localhost:3001",
        "http://labs.datahub.kr",
        "https://labs.datahub.kr",
        "http://localhost:8000",
        "https://localhost:8000",
    ]

@lru_cache()
def get_settings():
    return Settings() 

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str
    session_id: str = None

class FeedbackRequest(BaseModel):
    score: float
    run_id: str

current_user_id = None

# 기존 코드에서 수정이 필요한 부분만

@app.post("/api/")
async def stream_responses(request: Request):
    try:
        data = await request.json()
        message = data.get('message')
        client_session_id = data.get('session_id')
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # 🔧 핵심 수정: session_id가 없을 때만 새로 생성
        if not client_session_id:
            client_session_id = generate_session_id()
        else:
            pass

        config = RunnableConfig(
            recursion_limit=15, 
            configurable={
                "thread_id": "HIKE-JUSOCHATBOT-DEMO", 
                "user_id": current_user_id, 
                "session_id": client_session_id
            }
        )

        inputs = GraphState(
            question=message,
            session_id=client_session_id,
            q_type='',
            context='',
            answer='',
            relevance='',
        )

        try:
            final_state = graph.invoke(inputs, stream_mode="values", config=config)
            answer_text = final_state["answer"]
            
            # 응답에 현재 세션의 메시지 수 포함 (디버깅용)
            current_history = get_session_history(client_session_id)
            message_count = len(current_history.messages)
            
            print(f"💬 세션 {client_session_id[:8]}... 응답 완료 (총 {message_count}개 메시지)")
            
            return {
                "answer": answer_text,
                "session_id": client_session_id,  # 클라이언트가 다음에 사용할 수 있도록 반환
                "message_count": message_count
            }
            
        except GraphRecursionError as e:
            print(f"Recursion limit reached: {e}")
            return {
                "answer": "죄송합니다. 해당 질문에 대해서는 답변할 수 없습니다.",
                "session_id": client_session_id
            }
        except Exception as e:
            print(f"An error occurred: {e}")
            return {
                "answer": "죄송합니다. 처리 중 오류가 발생했습니다.",
                "session_id": client_session_id
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/reset")
async def reset_store(request: Request):
    global store, current_user_id
    
    try:
        data = await request.json()
        session_id_to_reset = data.get('session_id')
        
        if session_id_to_reset:
            # 특정 세션만 초기화
            if session_id_to_reset in store:
                message_count = len(store[session_id_to_reset].messages)
                del store[session_id_to_reset]
                print(f"🗑️ 세션 삭제: {session_id_to_reset[:8]}... ({message_count}개 메시지)")
            
            # 새로운 세션 ID 생성하여 반환
            new_session_id = generate_session_id()
            print(f"🆕 새 세션 생성: {new_session_id[:8]}...")
            
            return {
                "status": "Session reset successfully",
                "session_id": new_session_id,  # 새 세션 ID 반환
                "cleared_messages": message_count if session_id_to_reset in locals() else 0
            }
        else:
            # 모든 세션 초기화
            total_sessions = len(store)
            total_messages = sum(len(history.messages) for history in store.values())
            store = {}
            
            # 새로운 세션 ID 생성하여 반환
            new_session_id = generate_session_id()
            print(f"🧹 전체 초기화: {total_sessions}개 세션, {total_messages}개 메시지 삭제")
            
            return {
                "status": "All sessions reset successfully",
                "session_id": new_session_id,  # 새 세션 ID 반환
                "cleared_sessions": total_sessions,
                "cleared_messages": total_messages
            }
            
    except Exception as e:
        # 오류 발생시에도 새 세션 ID 반환
        store = {}
        new_session_id = generate_session_id()
        
        return {
            "status": "Sessions reset due to error",
            "session_id": new_session_id
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)