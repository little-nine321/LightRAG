import os
import logging
import asyncio

from lightrag import LightRAG, QueryParam
from lightrag.llm.zhipu import zhipu_complete, zhipu_embedding
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import json
from typing import List
from fastapi import File, UploadFile
import uuid
from datetime import datetime

WORKING_DIR = "./zsh/ragtest1/data"
UPLOAD_DIR = "./zsh/ragtest1/upload"
KV_STORE_DOC_STATUS_DIR = WORKING_DIR + "/kv_store_doc_status.json"

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
            func=lambda texts: zhipu_embedding(texts),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def insert_text_data_from_dir(rag: LightRAG, dir_path: str):

    rag = await initialize_rag()

    # 检查并创建file_status.json文件
    file_status_path = os.path.join(WORKING_DIR, "file_status.json")
    file_status = {}
    if not os.path.exists(file_status_path):
        with open(file_status_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    else:
        with open(file_status_path, "r", encoding="utf-8") as f:
            file_status = json.load(f)

    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(dir_path):
        for file in files:

            file_id = str(uuid.uuid4())
            file_path = os.path.join(root, file)

            # 更新文件状态
            file_status[file_id] = {
                "status": "processing",
                "created_at": datetime.now().isoformat(),
                "file_name": file,
                "updated_at": datetime.now().isoformat()
            }

            with open(file_status_path, "w", encoding="utf-8") as f:
                json.dump(file_status, f, ensure_ascii=False, indent=2) 

            # 插入文件内容, 这里默认是上传txt文件，后续做了文件解析可处理
            with open(file_path, 'r', encoding='utf-8') as f:
                await rag.ainsert(f.read(), ids=file_id)
            
            # 更新文件状态
            with open(KV_STORE_DOC_STATUS_DIR, "r", encoding="utf-8") as f:
                doc_status = json.load(f)
                file_status[file_id]["status"] = doc_status[file_id]["status"]
                file_status[file_id]["updated_at"] = datetime.now().isoformat()

            with open(file_status_path, "w", encoding="utf-8") as f:
                json.dump(file_status, f, ensure_ascii=False, indent=2)
    
    print("插入成功")

# 查询RAG
async def query_rag(rag: LightRAG | None, query: str, mode: str):
    if rag is None:
        rag = await initialize_rag()    
    message = await rag.aquery(query, param=QueryParam(mode=mode))
    return message

# 查询RAG流式
async def query_rag_stream(rag: LightRAG | None, query: str, mode: str):
    if rag is None:
        rag = await initialize_rag()    
    
    response = await rag.aquery(query, param=QueryParam(mode=mode, stream=True))

    async def stream_generator():
        if isinstance(response, str):
            # If it's a string, send it all at once
            yield response
        else:
            # If it's an async generator, send chunks one by one
            try:
                async for chunk in response:
                    if chunk:  # Only send non-empty content
                        yield chunk
            except Exception as e:
                logging.error(f"Streaming error: {str(e)}")
                yield f"{json.dumps({'error': str(e)})}\n"
    
    return stream_generator()

#上传文件
async def upload_document(rag: LightRAG | None, files: List[UploadFile]):
    if rag is None:
        rag = await initialize_rag()

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    # 检查并创建file_status.json文件
    file_status_path = os.path.join(WORKING_DIR, "file_status.json")
    file_status = {}
    if not os.path.exists(file_status_path):
        with open(file_status_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    else:
        with open(file_status_path, "r", encoding="utf-8") as f:
            file_status = json.load(f)
    
    flag = True

    for file in files:

        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, file_id)
        
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 更新文件状态
        file_status[file_id] = {
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "file_name": file.filename,
            "updated_at": datetime.now().isoformat()
        }

        with open(file_status_path, "w", encoding="utf-8") as f:
            json.dump(file_status, f, ensure_ascii=False, indent=2)

        try:
            # 插入文件内容, 这里默认是上传txt文件，后续做了文件解析可处理
            with open(file_path, "r", encoding="utf-8") as f:
                a = await rag.ainsert(f.read(), ids=file_id)
            
            # 更新文件状态
            with open(KV_STORE_DOC_STATUS_DIR, "r", encoding="utf-8") as f:
                doc_status = json.load(f)
                file_status[file_id]["status"] = doc_status[file_id]["status"]
                file_status[file_id]["updated_at"] = datetime.now().isoformat()
                if file_status[file_id]["status"] == "failed":
                    flag = False
                    
            with open(file_status_path, "w", encoding="utf-8") as f:
                json.dump(file_status, f, ensure_ascii=False, indent=2)
        except Exception as e:

            # 出错时更新状态为error
            file_status[file_id]["status"] = "failed"
            file_status[file_id]["updated_at"] = datetime.now().isoformat()
            
            with open(file_status_path, "w", encoding="utf-8") as f:
                json.dump(file_status, f, ensure_ascii=False, indent=2)
            
            flag = False
    
    return flag

# 查看文件列表
async def get_document():

    # 检查并创建file_status.json文件
    file_status_path = os.path.join(WORKING_DIR, "file_status.json")
    file_status = {}
    if not os.path.exists(file_status_path):
        with open(file_status_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    else:
        with open(file_status_path, "r", encoding="utf-8") as f:
            file_status = json.load(f)
    
            # 过滤出状态为processing或processed的文件
            valid_files = []
            for file_id, file_info in file_status.items():
                if file_info["status"] in ["processing", "processed"]:
                    valid_files.append({
                        "file_id": file_id,
                        "file_name": file_info["file_name"],
                        "created_at": file_info["created_at"], 
                        "updated_at": file_info["updated_at"]
                    })
            
            # 按照created_at降序排序,最新的排在前面
            valid_files.sort(key=lambda x: x["created_at"], reverse=True)
            return valid_files

# 删除文件
async def delete_document(rag: LightRAG | None, file_id: str):

    if rag is None:
        rag = await initialize_rag()
    
    # 检查并创建file_status.json文件
    file_status_path = os.path.join(WORKING_DIR, "file_status.json")
    file_status = {}
    if not os.path.exists(file_status_path):    
        with open(file_status_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    else:
        with open(file_status_path, "r", encoding="utf-8") as f:
            file_status = json.load(f)

    if file_id not in file_status.keys():
        return 1

    # 删除文件
    try:
        await rag.adelete_by_doc_id(file_id)

        file_status[file_id]["status"] = "deleted"
        file_status[file_id]["updated_at"] = datetime.now().isoformat()

        with open(file_status_path, "w", encoding="utf-8") as f:
            json.dump(file_status, f, ensure_ascii=False, indent=2)

        return 0
    except:
        return 2