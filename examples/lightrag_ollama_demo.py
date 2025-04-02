import asyncio
import nest_asyncio

nest_asyncio.apply()
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

import json

import datetime

WORKING_DIR = "./ragtest2"
CHROMADB_LOCAL_PATH = WORKING_DIR + "/chromadb"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:14b",
        llm_model_max_async=16,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="bge-m3:latest", host="http://localhost:11434"
            ),
        ),
        vector_storage="ChromaVectorDBStorage",
        vector_db_storage_cls_kwargs={
            "local_path": CHROMADB_LOCAL_PATH,
            "collection_settings": {
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 128,
                "hnsw:M": 16,
                "hnsw:batch_size": 50,
                "hnsw:sync_threshold": 500,
            },
        },
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


def main():

    start_time = datetime.datetime.now()

    with open("./ragtest2/test.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Insert example text
    with open("./input/新管控系统操作手册.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        data.insert(0, {"insertion_time": f"{duration:.3f}秒"})
        print("构建完成")

    for item in data:

        if data.index(item) == 0:
            continue

        question = item["question"]

        start_time = datetime.datetime.now()
        item["naive"] = rag.aquery(question, param=QueryParam(mode="naive"))
        end_time = datetime.datetime.now()
        item["naive_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

        start_time = datetime.datetime.now()
        item["local"] = rag.aquery(question, param=QueryParam(mode="local"))
        end_time = datetime.datetime.now()
        item["local_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

        start_time = datetime.datetime.now()
        item["global"] = rag.aquery(question, param=QueryParam(mode="global"))
        end_time = datetime.datetime.now()
        item["global_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

        start_time = datetime.datetime.now()
        item["hybrid"] = rag.aquery(question, param=QueryParam(mode="hybrid"))
        end_time = datetime.datetime.now()
        item["hybrid_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

        start_time = datetime.datetime.now()
        item["mix"] = rag.aquery(question, param=QueryParam(mode="mix"))
        end_time = datetime.datetime.now()
        item["mix_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

    with open("./ragtest2/test.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
