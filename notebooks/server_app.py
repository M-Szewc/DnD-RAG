import asyncio

# https://python.langchain.com/docs/langserve#server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder
from functools import partial
from operator import itemgetter

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from langchain_ollama import OllamaEmbeddings

from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb

from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel, Field
import ollama
from ollama import ChatResponse
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore


# Get environment variables

model_name = os.environ['OLLAMA_MODEL']
embed_model_name = os.environ['OLLAMA_EMBEDDING_MODEL']
ollama_address = os.environ['OLLAMA_ADDRESS']
ollama_port = os.environ['OLLAMA_PORT']

# Vector database

embed = OllamaEmbeddings(
    base_url=ollama_address+":"+ollama_port,
    model=embed_model_name
)

database_address = os.environ['IP_ADDRESS']
database_port = os.environ['DATABASE_PORT']

chroma_client = chromadb.HttpClient(host=database_address, port=database_port)
collection = chroma_client.get_or_create_collection(name="data")

vector_store_client = Chroma(
    client = chroma_client,
    collection_name="annotated_data",
    embedding_function=embed
)

filestore = LocalFileStore("./docstore")
docstore = create_kv_docstore(filestore)

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML, 
    chunk_size=400,
    chunk_overlap=80
)

retriever = ParentDocumentRetriever(
    vectorstore = vector_store_client,
    docstore=docstore,
    child_splitter=text_splitter
)

# define tool functions
def query_database(query: str, metadata : str = "") -> str:
    """
    Description of race or class that query needs to find

    Args:
        query (set): description of race or class that query needs to find
        metadata (set): The additional arguments to filter by e.g "race", "class" or "subclass"

    Returns:
        str: The fragment of homebrew
    """
    metadata = [term for term in metadata if term in ["race", "class", "subclass"]]
    print(f'Searching for: {query} with set metadata: {metadata}')

    retrieved_docs = []

    retrieved_docs = retriever.invoke(query)

    # Comment that out
    return retrieved_docs

    if len(metadata) > 0:
        retrieved_docs = vector_store_client.similarity_search(
            query,
            filter=lambda doc: doc.metadata.get("section") == metadata,
        )
    else:
        retrieved_docs = vector_store_client.similarity_search(
            query
        )

    return ["source:" + x.metadata["url"] + " content:" + x.page_content for x in retrieved_docs]

query_database_tool = {
    'type': 'function',
    'function': {
        'name': 'query_database',
        'description': 'Find race or class from DnD based on provided description',
        'parameters': {
            'type': 'object',
            'required': ['query', 'metadata'],
            'properties': {
                'query': {'type': 'string', 'description': 'description of race or class, their prefered characteristics and strengths'},
                'metadata': {'type': 'string', 'description': 'additional argument to filter e.g "race", "class" or "subclass"'},
            },
        },
    },
}

def recall_conversation(query: str) -> str:
    """
    Make a query about the current conversation

    Args:
        query (set): What to search for

    Returns:
        str: The fragment of conversation
    """

    return query



available_functions = {
    'query_database': query_database,
    'recall_conversation': recall_conversation,
}

async def generate_response(query):
    print(query)

    client = ollama.AsyncClient(host=ollama_address + ":" + ollama_port)

    messages = [{'role': 'user', 'content': query}]
    print('Prompt:', messages[0]['content'])

    response: ChatResponse = await client.chat(
        model_name,
        messages=messages,
        tools=[query_database_tool, recall_conversation]
    )

    output = dict()
    final_response = ""

    if response.message.tool_calls:
        # There may be multiple tool calls in the response
        for tool in response.message.tool_calls:
            # Ensure the function is available, and then call it
            if function_to_call:= available_functions.get(tool.function.name):
                print(f'Calling function:{tool.function.name}')
                print(f'Arguments:{tool.function.arguments}')
                output[tool.function.name] = function_to_call(**tool.function.arguments)
                print(f'Function output:{output[tool.function.name]}')
            else:
                print(f'Function {tool.function.name} not found')
    # Only need to chat with the model using the tool call results
    if response.message.tool_calls:
        # Add the function response to messages for the model to use
        messages.append(response.message)

        # There may be multiple tool calls
        for tool in response.message.tool_calls:
            # If the response if too long
            if len(output[tool.function.name]) > 1000:
                output[tool.function.name] = await get_summary(output[tool.function.name], query)
            # Add tool responses
            messages.append({'role': 'tool', 'content': str(output[tool.function.name]), 'tool_name': tool.function.name})

        # Get final response from model with function outputs
        final_response = await client.chat(model_name, messages=messages)
        print(f'Final response: {final_response.message.content}')
    else:
        print(f'No tool calls returned from model')
        final_response = await client.chat(model_name, messages=messages)
        print(f'Final response: {final_response.message.content}')

    return final_response

def sync_gen(query):
    response = asyncio.run(generate_response(query))
    print(response["message"]["content"])
    return response["message"]["content"]

# Routes
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_routes(
    app,
    RunnableLambda(lambda x: invoke(x)),
    path="/basic_chat",
)

add_routes(
    app,
    RunnableLambda(lambda x: sync_gen(x)),
    path="/generator",
)

add_routes(
    app,
    RunnableLambda(lambda x: vector_store_client.as_retriever(x)),
    path="/retriever",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5678)

#await generate_response("Hi")
