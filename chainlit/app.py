import os
from typing import List
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from chainlit.types import AskFileResponse
import chainlit as cl

import sys
sys.path.append('..')
from policy_rag.vectorstore_utils import QdrantVectorstoreHelper
from policy_rag.chains import get_qa_chain
from policy_rag.app_utils import get_embedding_model



QDRANT_COLLECTION = 'policy-embed-te3-large-plus'
VECTORSTORE_MODEL = {
    'model_source': 'openai',
    'model_name': 'text-embedding-3-large',
    'vector_size': 3072
}
K = 5



@cl.on_chat_start
async def on_chat_start():
    qdrant_retriever = QdrantVectorstoreHelper()\
        .get_retriever(
            collection_name=QDRANT_COLLECTION,
            embedding_model=get_embedding_model(VECTORSTORE_MODEL),
            k=K
        )
    
    rag_qa_chain = get_qa_chain(retriever=qdrant_retriever, streaming=True)
    rag_qa_chain = rag_qa_chain   

    cl.user_session.set("rag_qa_chain", rag_qa_chain)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="AI Policy Basics",
            message="What are the principles of AI safety?"
        )
    ]


@cl.on_message
async def main(message):
    rag_qa_chain = cl.user_session.get("rag_qa_chain")

    msg = cl.Message(content="", elements=[])
    async for chunk in rag_qa_chain.astream({"question": message.content}):
        print(chunk)
        if contexts := chunk.get("contexts"):
            source_elements = [
                cl.Text(content=f"Page {doc.metadata['page']}", name=doc.metadata['title'], display="inline") for doc in contexts
            ]
        if answer := chunk.get("answer"):
            await msg.stream_token(answer.content)

    msg.elements = source_elements
    await msg.send()
    