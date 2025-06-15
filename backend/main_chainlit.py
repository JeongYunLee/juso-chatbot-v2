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
# from langchain_teddynote.messages import random_uuid   # type: ignore
## langsmith
from langsmith import Client
from langchain_teddynote import logging
from langchain_core.tracers.context import collect_runs

from langgraph.graph import END, StateGraph # type: ignore
from langgraph.checkpoint.memory import MemorySaver # type: ignore
from langgraph.errors import GraphRecursionError # type: ignore

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

## Flask
from flask import Flask, request, jsonify, g, session
from flask_restx import Api, Resource
from flask_cors import CORS

from langchain_teddynote import logging

# .env 파일 활성화 & API KEY 설정
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
logging.langsmith("rag_chatbot_test")

llm_4o = ChatOpenAI(model="gpt-4o", temperature=0)

class GraphState(TypedDict):
    question: str # 질문
    q_type: str  # 질문의 유형
    context: list | str  # 문서의 검색 결과
    answer: str | list[str]   # llm이 생성한 답변
    relevance: str  # 답변의 문서에 대한 관련성 (groundness check)
    # chat_history: str  # 채팅 히스토리
    
store = {}

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

#######################################################################
############################ nodes: Router ############################
#######################################################################

# 자료구조 정의 (pydantic)
class Router(BaseModel):
    type: str = Field(description="type of the query that model choose")

# 출력 파서 정의
router_output_parser = JsonOutputParser(pydantic_object=Router)
format_instructions = router_output_parser.get_format_instructions()

# prompt 구성
router_prompt = PromptTemplate(
    template="""
            You are an expert who classifies the type of question. There are two query types: [‘general’, ‘domain_specific’]

            [general]
            Questions unrelated to addresses, such as translating English to Korean, asking for general knowledge (e.g., “What is the capital of South Korea?”), or queries that can be answered through a web search.

            [domain_specific]
            Questions related to addresses, such as concepts, definitions, address-related data analysis, or reviewing properly written addresses (e.g., “수지구는 자치구이니 일반구이니?”, “특별시에 대해서 설명해줘”, “주소와 주소정보의 차이점은?”).

            <Output format>: Always respond with either “general” or “domain_specific” and nothing else. {format_instructions}
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
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="query",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    
    router_result = router_with_history.invoke({"query": state["question"]})
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
    retrieved_docs = vectorstore.similarity_search_with_score(state["question"], k=3)
    return {**state, "context": retrieved_docs} 

#########################################################################
############################ nodes: Verifier ############################
#########################################################################

# 자료구조 정의 (pydantic)
class Verifier(BaseModel):
    type: str = Field(description="verify that retrieved data is sufficient to answer the query")

# 출력 파서 정의
verifier_output_parser = JsonOutputParser(pydantic_object=Verifier)
format_instructions = verifier_output_parser.get_format_instructions()

verifier_prompt = PromptTemplate(
    template="""
            You are an expert who verity the retrieved data's quality and usefullness to answer the query. There are two query types: [‘sufficient’, ‘insufficient’, 'unsuitable']

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
    verified = chain.invoke({"query": state["question"], "retrieved_data": state["context"]})
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

# DallE API Wrapper를 생성합니다.
dalle = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="standard", n=1)


# DallE API Wrapper를 도구로 정의합니다.
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

    # llm_4_1 = ChatOpenAI(model="gpt-4.1")

    # model_prompt = PromptTemplate(
    #     template="""
    #         You are an expert who answers the query based on the retrieved data.
    #         When relevance is `sufficient`, you MUST INCLUDE the sources from the retrieved_data.
    #         Follow these rules for citing sources:
    #             1.	Only include a source if it was actually used in your answer and relevance is sufficient.
    #             2.	The citation must be based on context/metadata/source in the retrieved data(3 context would be provided). DO NOT MAKE YOURSELF!
    #             3.	Citation format:
    #             •	If context/metadata/source is: data/final/[1018] 주소정보_업무편람_최종(하이퍼링크).docx
    #             •	Then citation should be: 출처: 주소정보 업무편람 최종
    #             •	Extract only the essential name (remove path, numbering, and file extensions/parentheses).
    #         DON'T FORGET TO INCLUDE THE REFERENCE INFO AT THE BOTTOM OF THE ANSWER!
    #         <Question>: {query}
    #         <Retrieved data>: {retrieved_data}
    #     """,
    #     input_variables=["query", "retrieved_data", "chat_history"],
    # )

    # chain = model_prompt | llm_4_1 | StrOutputParser()
    # return chain.invoke({"query": input, "retrieved_data": retrieved_data})

@tool
def image_explainer(query, image_url):
    """ 이미지 설명 생성기. Use after the image_generator tool make the image output. Use the url of the image_generator tool. """
    client = OpenAI()  # Client 객체 생성

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

            # """
            # When using `Retrieved Data` and the topic is address-related, follow these rules for citing sources:
            #     1.	Only include a source if it was actually used in your answer and relevance is sufficient.
            #     2.	The citation must be based on context/metadata/source in the retrieved data. Among the three provided context sources, cite only those directly used in your answer.
            #     3.	Citation format:
            #     •	If context/metadata/source is: data/final/********.docx
            #     •	Then citation should be: 출처: **********
            #     •	Extract only the essential name (remove path, numbering, and file extensions/parentheses).
            #     (IMPORTANT) DO NOT MAKE UP SOURCES. Only use the sources provided in the retrieved_data.
            # Additional rules:
            # 1.	You can appropriately select and use various tools based on the situation.
            #     (Note) When the user asks for current information, consider "June 2025" as the reference point for up-to-date data.
            # 2.	If the relevance is "insufficient," use the retrieved_data along with the search_on_web tool.
            # 3.	If you use the advanced_assistant tool, make sure to reflect both the content and the full extent of the response in your final answer.
            # 4.	If you use the image_generator tool, you must also use the image_explainer tool to provide a description of the generated image.
            # 5.	(IMPORTANT) The user may have multiple requests, and you must generate a response that addresses all of them.
            #     - If multiple tools are required to fulfill different aspects of the user's request, combine the results comprehensively into a single, final response.
            #     - If an image is generated, you must include both the image URL and a detailed explanation of the image.
            #     - Ensure that all tool outputs are fully incorporated into the final answer.
            # """
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
    
    # tool_input = {"input": state["question"], "retrieved_data": state["context"], "relevance": state["relevance"]}


    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,  # 세션 기록을 가져오는 함수
        # input_messages_key="input",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )


    result = agent_with_history.invoke({"input": state["question"], "retrieved_data": state["context"], "relevance": state["relevance"]})
    state['answer'] = result['output']

    return state

########################################################################
############################ Workflow Graph ############################
########################################################################

workflow = StateGraph(GraphState)

# 노드들을 정의합니다.
workflow.add_node("Router", router)  # 질문의 종류를 분류하는 노드를 추가합니다.
workflow.add_node("Retrieved Data", retrieve_document)  # 답변을 검색해오는 노드를 추가합니다.
workflow.add_node("Agent", agent)  # 일반 질문에 대한 답변을 생성하는 노드를 추가합니다.
workflow.add_node("Verifier", verifier)  # 답변의 문서에 대한 관련성 체크 노드를 추가합니다.
# workflow.add_node("llm_answer", llm_model)  # 답변을 생성하는 노드를 추가합니다.

# 조건부 엣지를 추가합니다.
workflow.add_conditional_edges(
    "Router",  # 질문의 종류를 분류하는 노드에서 나온 결과를 기반으로 다음 노드를 선택합니다.
    router_conditional_edge,
    {"domain_specific": "Retrieved Data",  "general": "Agent"},
)

workflow.add_edge("Retrieved Data", "Verifier")  # 검색 -> 답변
workflow.add_edge("Verifier", "Agent")  # 답변 -> 답변

workflow.add_edge("Agent", END)  # 답변 -> 종료

# 시작점을 설정합니다.
workflow.set_entry_point("Router")

# 기록을 위한 메모리 저장소를 설정합니다.
memory = MemorySaver()

# 그래프를 컴파일합니다.
graph = workflow.compile(checkpointer=memory)    

##############################################################################################################
################################################Chat Interface################################################
##############################################################################################################

# app = Flask(__name__)
# app.secret_key = os.getenv('FLASK_SECRET_KEY')

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

current_user_id = None

@app.post("/set_user_id")
async def set_user_id(request: Request):
    global current_user_id
    data = await request.json()
    current_user_id = data.get('id')
    return {"status": "ID received successfully"}

@app.post("/")
async def stream_responses(request: Request):
    try:
        data = await request.json()
        message = data.get('message')
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        config = RunnableConfig(
            recursion_limit=15, 
            configurable={
                "thread_id": "HIKE-JUSOCHATBOT-DEMO", 
                "user_id": current_user_id, 
                "session_id": "session1"
            }
        )

        inputs = GraphState(
            question=message,
        )

        try:
            final_state = graph.invoke(inputs, stream_mode="values", config=config)
            answer_text = final_state["answer"]
            print(answer_text)
            return {"answer": answer_text}
            
        except GraphRecursionError as e:
            print(f"Recursion limit reached: {e}")
            return {"answer": "죄송합니다. 해당 질문에 대해서는 답변할 수 없습니다."}
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"answer": "죄송합니다. 처리 중 오류가 발생했습니다."}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/feedback")
async def handle_feedback(feedback: FeedbackRequest):
    try:
        langsmith_client.create_feedback(
            feedback.run_id,
            key="feedback-key",
            score=feedback.score,
            comment="comment",
        )
        return {"message": "Feedback received"}
    except Exception as e:
        print(f"An error occurred while handling feedback: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing feedback")

@app.post("/reset")
async def reset_store():
    global store, current_user_id
    store = {}
    
    global initial_state
    initial_state = GraphState(
        question='',
        q_type='',
        context='',
        answer='',
        relevance='',
    )
    
    return {"status": "Store and GraphState reset successfully"}

# if __name__ == '__main__':
#     app.run('0.0.0.0', port=5000, debug=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_chainlit:app", host="0.0.0.0", port=8000, reload=True)