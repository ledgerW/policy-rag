import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents.base import Document
from policy_rag.data_models import DocList

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from replit.object_storage import Client


# Text Loading
class DocLoader:
    docs: DocList = DocList([]).root

    def load(self, path: str) -> List[Document]:
        if path.endswith('.pdf'):
            loader = PyMuPDFLoader(path)
            self.docs.extend(loader.load())
        else:
            print(f'Skipping {path} - not PDF')

        return self.docs

    
    def load_dir(self, dir_path: str) -> List[Document]:
        for doc_name in os.listdir(dir_path):
            doc_path = os.path.join(dir_path, doc_name)
            self.load(doc_path)

        return self.docs

    
    def upload_to_storage(self, file_path: str, file_name: str):
        client = Client()
        with open(file_path, 'rb') as file:
            client.upload_from_file(file_name, file)
    
    
    def download_from_storage(self, file_name: str, dest_path: str):
        client = Client()
        obj = client.download(file_name)
        with open(dest_path, 'wb') as file:
            file.write(obj)
    

# Text Splitting
def get_recursive_token_chunks(
        docs: List[Document],
        model_name: str = 'gpt-4',
        chunk_size: int = 150,
        chunk_overlap: int = 0
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return text_splitter.split_documents(docs)


def get_semantic_chunks(
        docs: List[Document],
        embedding_model: OpenAIEmbeddings,
        breakpoint_type: str = 'gradient'
) -> List[Document]:
    text_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type=breakpoint_type
    )

    return text_splitter.split_documents(docs)