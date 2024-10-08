import os
from dotenv import load_dotenv
load_dotenv()

from typing import Dict, Tuple
from collections.abc import Callable

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from policy_rag.text_utils import get_recursive_token_chunks, get_semantic_chunks



# Config Options
CHUNK_METHOD = {
    'token-overlap': get_recursive_token_chunks,
    'semantic': get_semantic_chunks
}

EMBEDDING_MODEL_SOURCE = {
    'openai': OpenAIEmbeddings,
    'huggingface': HuggingFaceInferenceAPIEmbeddings
}


# Helpers
def get_chunk_func(chunk_method: Dict) -> Tuple[Callable, Dict]:
    chunk_func = CHUNK_METHOD[chunk_method['method']]

    if chunk_method['method'] == 'token-overlap':
        chunk_func_args = chunk_method['args']
    
    if chunk_method['method'] == 'semantic':
        args = chunk_method['args']
        chunk_func_args = {
            'embedding_model': EMBEDDING_MODEL_SOURCE[args['model_source']](model=args['model_name']),
            'breakpoint_type': args['breakpoint_type']
        }

    return chunk_func, chunk_func_args


def get_embedding_model(config) -> OpenAIEmbeddings | HuggingFaceInferenceAPIEmbeddings:
    if config['model_source'] == 'openai':
        model = EMBEDDING_MODEL_SOURCE[config['model_source']](model=config['model_name'])

    if config['model_source'] == 'huggingface':
        model = EMBEDDING_MODEL_SOURCE[config['model_source']](
            api_key=os.getenv('HF_API_KEY'),
            model_name=config['model_name'],
            api_url=config['api_url']
        )

    return model