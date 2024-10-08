import os
from dotenv import load_dotenv
load_dotenv()

import yaml
import argparse


from policy_rag.text_utils import DocLoader
from policy_rag.vectorstore_utils import QdrantVectorstoreHelper
from policy_rag.app_utils import (
    CHUNK_METHOD,
    EMBEDDING_MODEL_SOURCE,
    get_chunk_func,
    get_embedding_model
)
from policy_rag.eval_utils import eval_on_ls_dataset







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='YAML config file to run')
    parser.add_argument('--config_dir', default=None, help='Directory of YAML config files to run')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config_yml = yaml.safe_load(file)

    data_dir = config_yml['data_dir']
    chunk_method = config_yml['chunk_method']
    ls_project = config_yml['ls_project']
    ls_dataset_name = config_yml['ls_dataset_name']
    ls_experiment_name = config_yml['ls_experiment_name']
    vectorstore_model = config_yml['vectorstore_model']
    metrics = config_yml['metrics']

    os.environ['LANGCHAIN_PROJECT'] = ls_project
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'


    # Load Raw Data
    print('Loading Docs')
    loader = DocLoader()
    docs = loader.load_dir(data_dir)


    # Chunk Docs
    print('Chunking Docs')
    chunk_func, chunk_func_args = get_chunk_func(chunk_method)
    print(chunk_func_args)
    chunks = chunk_func(docs=docs, **chunk_func_args)
    print(f"len of chunks: {len(chunks)}")


    # Load chunks into vectorstore
    print('Creating Qdrant Collection and Getting Retriever')
    qdrant_vectorstore = QdrantVectorstoreHelper()
    qdrant_vectorstore.create_cloud_vectorstore(
        chunks=chunks,
        collection_name=ls_experiment_name,
        embedding_model=get_embedding_model(vectorstore_model),
        vector_size=vectorstore_model['vector_size']
    )
    retriever = qdrant_vectorstore.get_retriever(
        collection_name=ls_experiment_name,
        embedding_model=get_embedding_model(vectorstore_model),
        k=3
    )
    
    # Run RAGAS Evaluation in LangSmith
    result = eval_on_ls_dataset(
        metrics=metrics,
        retriever=retriever,
        ls_dataset_name=ls_dataset_name,
        ls_project_name=ls_project,
        ls_experiment_name=ls_experiment_name
    )