# 개선된 서버 코드 (main.py)

import uuid, os
import asyncio
import threading
from contextlib import asynccontextmanager
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

# 🔧 개선 1: 스레드 안전한 저장소
import threading
from collections import defaultdict

class ThreadSafeStore:
    def __init__(self):
        self._store = {}
        self._lock = threading.RLock()  # 재진입 가능한 락
    
    def get_session_history(self, session_id: str):
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ChatMessageHistory()
                print(f"🆕 새로운 세션 히스토리 생성: {session_id[:8]}...")
            return self._store[session_id]
    
    def clear_session(self, session_id: str = None):
        with self._lock:
            if session_id:
                if session_id in self._store:
                    message_count = len(self._store[session_id].messages)
                    del self._store[session_id]
                    return message_count
                return 0
            else:
                total_sessions = len(self._store)
                total_messages = sum(len(history.messages) for history in self._store.values())
                self._store.clear()
                return total_sessions, total_messages
    
    def get_stats(self):
        with self._lock:
            return {
                'total_sessions': len(self._store),
                'total_messages': sum(len(history.messages) for history in self._store.values())
            }

# 전역 스레드 안전 저장소
thread_safe_store = ThreadSafeStore()

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    return thread_safe_store.get_session_history(session_ids)

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

# 🔧 개선 2: ChromaDB 연결 풀링 및 재시도 로직
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        print(f"⚠️ 시도 {attempt + 1} 실패, {delay}초 후 재시도: {str(e)[:100]}")
                        time.sleep(delay * (attempt + 1))  # 지수 백오프
                    else:
                        print(f"❌ 모든 재시도 실패: {str(e)}")
            raise last_exception
        return wrapper
    return decorator

# ChromaDB 클라이언트를 함수 내부에서 생성하여 충돌 방지
@retry_on_failure(max_retries=3, delay=1)
def get_vectorstore():
    try:
        client = chromadb.PersistentClient('chroma/')
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')  
        vectorstore = Chroma(client=client, collection_name="49_files_openai_3072", embedding_function=embedding)
        return vectorstore
    except Exception as e:
        print(f"ChromaDB 연결 실패: {e}")
        raise

def retrieve_document(state: GraphState) -> GraphState:
    try:
        vectorstore = get_vectorstore()
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
    except Exception as e:
        print(f"문서 검색 실패: {e}")
        # 빈 컨텍스트로 계속 진행
        return {**state, "context": []}

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

# 🔧 개선 3: OpenAI API 레이트 리미팅 및 재시도
import openai
from openai import RateLimitError, APITimeoutError

@retry_on_failure(max_retries=3, delay=2)
def call_openai_with_retry(client, **kwargs):
    try:
        return client.chat.completions.create(**kwargs)
    except RateLimitError as e:
        print(f"⚠️ OpenAI 레이트 리미트: {e}")
        time.sleep(5)  # 레이트 리미트 시 더 오래 대기
        raise
    except APITimeoutError as e:
        print(f"⚠️ OpenAI 타임아웃: {e}")
        raise
    except Exception as e:
        print(f"⚠️ OpenAI API 오류: {e}")
        raise

@tool
def search_on_web(input):
    """ 실시간 정보, 최신 정보 등 웹 검색이 필요한 질문에 답변하기 위해 사용하는 도구 """
    try:
        search_tool = TavilySearchResults(max_results=5)
        search_result = search_tool.invoke({"query": input})
        return search_result
    except Exception as e:
        print(f"웹 검색 실패: {e}")
        return "웹 검색 중 오류가 발생했습니다."

from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.tools import tool

dalle = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="standard", n=1)

@tool 
def dalle_tool(query):
    """use this tool to generate image from text"""
    try:
        return dalle.run(query)
    except Exception as e:
        print(f"이미지 생성 실패: {e}")
        return "이미지 생성 중 오류가 발생했습니다."

@tool
def advanced_assistant(input, retrieved_data):
    """ 고급 기능(예: 보고서 생성, 긴 문서 생성, 추론이 필요한 답변 등)을 수행하기 위한 모델 """
    try:
        client = OpenAI()
        
        response = call_openai_with_retry(
            client,
            model="gpt-4o",  # o3 대신 더 안정적인 gpt-4o 사용
            messages=[
                { "role": "developer", "content": "You are a helpful assistant." },
                {
                    "role": "user", 
                    "content": f"query: {input}\n\n retrieved data: {retrieved_data}"
                }
            ],
            timeout=60  # 타임아웃 설정
        )
        
        result = response.choices[0].message.content
        return result
    except Exception as e:
        print(f"고급 지원 실패: {e}")
        return "고급 기능 처리 중 오류가 발생했습니다."

@tool
def image_explainer(query, image_url):
    """ 이미지 설명 생성기. Use after the image_generator tool make the image output. Use the url of the image_generator tool. """
    try:
        client = OpenAI()

        response = call_openai_with_retry(
            client,
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
            timeout=60
        )
        return response.choices[0]
    except Exception as e:
        print(f"이미지 설명 실패: {e}")
        return "이미지 설명 생성 중 오류가 발생했습니다."
    
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
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60)  # 타임아웃 설정
        agent = create_tool_calling_agent(llm, tools, agent_prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=10,  # 반복 횟수 줄임
            max_execution_time=120,  # 실행 시간 제한
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
    except Exception as e:
        print(f"에이전트 실행 실패: {e}")
        state['answer'] = f"죄송합니다. 질문 처리 중 오류가 발생했습니다: {str(e)[:100]}"
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

# 🔧 개선 4: FastAPI 앱 생성 시 lifespan 이벤트 추가
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시
    print("🚀 서버 시작")
    yield
    # 종료 시
    print("🛑 서버 종료")

app = FastAPI(title="Juso Chatbot API", lifespan=lifespan)

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

# 🔧 개선 5: 비동기 처리 및 상세한 에러 핸들링
@app.post("/api/")
async def stream_responses(request: Request):
    try:
        data = await request.json()
        message = data.get('message')
        client_session_id = data.get('session_id')
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        if not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # 메시지 길이 제한
        if len(message) > 1000:
            raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")

        # 세션 ID 처리
        if not client_session_id:
            client_session_id = generate_session_id()

        # 🔧 개선 6: 설정 최적화
        config = RunnableConfig(
            recursion_limit=10,  # 재귀 제한 줄임
            configurable={
                "thread_id": f"HIKE-JUSOCHATBOT-{client_session_id[:8]}", 
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
            # 타임아웃 설정으로 무한 대기 방지
            final_state = await asyncio.wait_for(
                asyncio.to_thread(graph.invoke, inputs, config),
                timeout=180  # 3분 타임아웃
            )
            
            answer_text = final_state["answer"]
            
            # 응답 검증
            if not answer_text or not isinstance(answer_text, str):
                answer_text = "죄송합니다. 응답을 생성할 수 없습니다."
            
            # 세션 통계
            current_history = get_session_history(client_session_id)
            message_count = len(current_history.messages)
            
            print(f"✅ 세션 {client_session_id[:8]}... 응답 완료 (총 {message_count}개 메시지)")
            
            return {
                "answer": answer_text,
                "session_id": client_session_id,
                "message_count": message_count,
                "status": "success"
            }
            
        except asyncio.TimeoutError:
            print(f"⏰ 타임아웃: {client_session_id[:8]}...")
            return {
                "answer": "죄송합니다. 응답 시간이 초과되었습니다. 다시 시도해 주세요.",
                "session_id": client_session_id,
                "status": "timeout"
            }
        except GraphRecursionError as e:
            print(f"🔄 재귀 제한 초과: {e}")
            return {
                "answer": "죄송합니다. 질문이 너무 복잡합니다. 더 간단한 질문으로 다시 시도해 주세요.",
                "session_id": client_session_id,
                "status": "recursion_error"
            }
        except Exception as e:
            print(f"❌ 그래프 실행 오류: {type(e).__name__}: {str(e)[:200]}")
            return {
                "answer": "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                "session_id": client_session_id,
                "status": "error",
                "error_type": type(e).__name__
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API 오류: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/reset")
async def reset_store(request: Request):
    try:
        data = await request.json()
        session_id_to_reset = data.get('session_id')
        
        if session_id_to_reset:
            # 특정 세션만 초기화
            message_count = thread_safe_store.clear_session(session_id_to_reset)
            new_session_id = generate_session_id()
            
            print(f"🗑️ 세션 삭제: {session_id_to_reset[:8]}... ({message_count}개 메시지)")
            
            return {
                "status": "Session reset successfully",
                "session_id": new_session_id,
                "cleared_messages": message_count
            }
        else:
            # 모든 세션 초기화
            total_sessions, total_messages = thread_safe_store.clear_session()
            new_session_id = generate_session_id()
            
            print(f"🧹 전체 초기화: {total_sessions}개 세션, {total_messages}개 메시지 삭제")
            
            return {
                "status": "All sessions reset successfully",
                "session_id": new_session_id,
                "cleared_sessions": total_sessions,
                "cleared_messages": total_messages
            }
            
    except Exception as e:
        print(f"❌ 리셋 오류: {e}")
        # 오류 발생시에도 새 세션 ID 반환
        new_session_id = generate_session_id()
        
        return {
            "status": "Sessions reset due to error",
            "session_id": new_session_id,
            "error": str(e)
        }

# 🔧 개선 7: 헬스체크 엔드포인트 추가
@app.get("/health")
async def health_check():
    stats = thread_safe_store.get_stats()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "sessions": stats['total_sessions'],
        "messages": stats['total_messages']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        workers=1,  # 단일 워커로 메모리 공유 문제 방지
        timeout_keep_alive=30,
        limit_concurrency=100,  # 동시 연결 제한
        limit_max_requests=1000  # 최대 요청 수 제한
    )