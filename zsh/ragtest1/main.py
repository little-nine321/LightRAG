import os
import logging
import asyncio

from lightrag import LightRAG, QueryParam
from lightrag.llm.zhipu import zhipu_complete, zhipu_embedding
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./zsh/ragtest1/data"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "qwen2.5:14b",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="sk-1234567890",
        base_url="http://localhost:11434/v1",
        **kwargs,
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_max_async=10,
        llm_model_max_token_size=32768,
        embedding_func=EmbeddingFunc(
            embedding_dim=2048,  # Zhipu embedding-3 dimension
            max_token_size=8192,
            func=lambda texts: zhipu_embedding(texts,api_key="1e54498308567d4a66a210ea84420441.ieipu9oRxO8qO7W8"),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    # Initialize RAG instance
    rag = await initialize_rag()

    with open("./zsh/ragtest1/input/story.txt", "r", encoding="utf-8") as f:
        await rag.ainsert(f.read())

    # Perform local search
    print(await rag.aquery(
            "总结该故事?", param=QueryParam(mode="local", stream=True)
    ))

    # await rag.adelete_by_doc_id("doc-0e360941839d2f772c01c35c18f088d8")

    # print("删除成功")


if __name__ == "__main__":
    asyncio.run(main())