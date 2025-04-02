import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status
import json

import datetime

WORKING_DIR = "./ragtest"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "glm-4-air",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="embedding-3",
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)

# asyncio.run(test_funcs())


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:

        start_time = datetime.datetime.now()

        # Initialize RAG instance
        rag = await initialize_rag()

        with open("./ragtest/test.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        with open("./input/新管控系统操作手册.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            data.insert(0, {"insertion_time": f"{duration:.3f}秒"})
            print("构建完成")
        


        for item in data:

            if data.index(item) == 0:
                continue

            question = item["question"]

            start_time = datetime.datetime.now()
            item["naive"] = await rag.aquery(question, param=QueryParam(mode="naive"))
            end_time = datetime.datetime.now()
            item["naive_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

            start_time = datetime.datetime.now()
            item["local"] = await rag.aquery(question, param=QueryParam(mode="local"))
            end_time = datetime.datetime.now()
            item["local_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

            start_time = datetime.datetime.now()
            item["global"] = await rag.aquery(question, param=QueryParam(mode="global"))
            end_time = datetime.datetime.now()
            item["global_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

            start_time = datetime.datetime.now()
            item["hybrid"] = await rag.aquery(question, param=QueryParam(mode="hybrid"))
            end_time = datetime.datetime.now()
            item["hybrid_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

            start_time = datetime.datetime.now()
            item["mix"] = await rag.aquery(question, param=QueryParam(mode="mix"))
            end_time = datetime.datetime.now()
            item["mix_query_time"] = f"{(end_time - start_time).total_seconds():.3f}秒"

        with open("./ragtest/test.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())