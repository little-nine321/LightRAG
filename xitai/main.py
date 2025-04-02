import os
import logging
import asyncio
import json
import datetime

from lightrag import LightRAG, QueryParam
from lightrag.llm.zhipu import zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./xitai/ragtest"

CHROMADB_LOCAL_PATH = "./xitai/ragtest/chromadb"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("Please set ZHIPU_API_KEY in your environment")


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=zhipu_complete,
        llm_model_name="glm-4-air",  # Using the most cost/performance balance model, but you can change it here.
        llm_model_max_async=50,
        llm_model_max_token_size=32768,
        embedding_func=EmbeddingFunc(
            embedding_dim=2048,  # Zhipu embedding-3 dimension
            max_token_size=8192,
            func=lambda texts: zhipu_embedding(texts, api_key="1e54498308567d4a66a210ea84420441.ieipu9oRxO8qO7W8"),
        ),
        vector_storage="ChromaVectorDBStorage",
        vector_db_storage_cls_kwargs={
            "local_path": CHROMADB_LOCAL_PATH,
            "collection_settings": {
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 128,
                "hnsw:M": 16,
                "hnsw:batch_size": 100,
                "hnsw:sync_threshold": 1000,
            },
        },
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def main():

    # start_time = datetime.datetime.now()

    rag = await initialize_rag()

    # input_dir = "./xitai/input"
    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".txt"):
    #         with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
    #             await rag.ainsert(f.read())

    # end_time = datetime.datetime.now()
    # duration = (end_time - start_time).total_seconds()
    # answer_list.append({"build_index_time": f"{duration:.3f}秒", "index":0})

    # with open("./xitai/ask/answer.json", "w", encoding="utf-8") as f:
    #     json.dump(answer_list, f, ensure_ascii=False, indent=4)
    
    # print("构建完成")

    # 开始提问
    # with open("./xitai/ask2/qa.json", "r", encoding="utf-8") as f:
    #     qas = json.load(f)

    # for index, question in enumerate(qas):
        
    #     now_index = qas[0]["ask_index"]

    #     if index < now_index:
    #         continue

    #     start_time = datetime.datetime.now()
    #     question["naive_answer "] = await rag.aquery(question["question"], param=QueryParam(mode="naive"))
    #     end_time = datetime.datetime.now()
    #     duration = (end_time - start_time).total_seconds()
    #     question["naive_time"] =  f"{duration:.3f}秒"

    #     start_time = datetime.datetime.now()
    #     question["local_answer"] = await rag.aquery(question["question"], param=QueryParam(mode="local"))
    #     end_time = datetime.datetime.now()
    #     duration = (end_time - start_time).total_seconds()
    #     question["local_time"] =  f"{duration:.3f}秒"

    #     start_time = datetime.datetime.now()
    #     question["global_answer"] = await rag.aquery(question["question"], param=QueryParam(mode="global"))
    #     end_time = datetime.datetime.now()
    #     duration = (end_time - start_time).total_seconds()
    #     question["global_time"] =  f"{duration:.3f}秒"

    #     start_time = datetime.datetime.now()
    #     question["hybrid_answer"] = await rag.aquery(question["question"], param=QueryParam(mode="hybrid"))
    #     end_time = datetime.datetime.now()
    #     duration = (end_time - start_time).total_seconds()
    #     question["hybrid_time"] =  f"{duration:.3f}秒"

    #     start_time = datetime.datetime.now()
    #     question["mix_answer"] = await rag.aquery(question["question"], param=QueryParam(mode="mix"))
    #     end_time = datetime.datetime.now()
    #     duration = (end_time - start_time).total_seconds()
    #     question["mix_time"] =  f"{duration:.3f}秒"

    #     now_index += 1
    #     qas[0]["ask_index"] = now_index

    #     with open("./xitai/ask2/qa.json", "w", encoding="utf-8") as f:
    #         json.dump(qas, f, ensure_ascii=False, indent=4)

    #     print(f"第{index}题完成")

    while True:
        question = input("请输入问题：")
        answer = await rag.aquery(question, param=QueryParam(mode="naive"))
        print(answer)

if __name__ == "__main__":
    asyncio.run(main())