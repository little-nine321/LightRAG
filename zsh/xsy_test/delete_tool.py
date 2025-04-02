#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightRAG删除工具 - 根据实体名或文档ID删除数据
"""

import os
import sys
import argparse
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
import nest_asyncio

# 应用nest_asyncio解决事件循环嵌套问题
nest_asyncio.apply()

# 默认配置
DEFAULT_RAG_DIR = "index_default"
WORKING_DIR = os.environ.get("RAG_DIR", DEFAULT_RAG_DIR)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-14b-instruct")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-bge-m3")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:12345/v1")
API_KEY = os.environ.get("API_KEY", "None")


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=BASE_URL,
        api_key=API_KEY,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


def initialize_rag():
    """初始化LightRAG实例"""
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        
    # 初始化RAG实例
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=asyncio.run(get_embedding_dim()),
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
        ),
    )
    return rag


async def verify_entity_deletion(rag, entity_name):
    """验证实体是否已成功删除"""
    entity_name_upper = f'"{entity_name.upper()}"'
    
    # 检查实体是否仍存在于图中
    if entity_name_upper in rag.chunk_entity_relation_graph._graph.nodes():
        print(f"警告: 实体 '{entity_name}' 仍存在于关系图中")
        return False
    
    # 检查实体是否仍存在于向量数据库中
    try:
        entities_data = rag.entities_vdb.client_storage.get("data", [])
        for entity in entities_data:
            if entity.get("entity_name") == entity_name_upper:
                print(f"警告: 实体 '{entity_name}' 仍存在于向量数据库中")
                return False
    except:
        print("无法验证实体向量数据库")
    
    return True


async def verify_doc_deletion(rag, doc_id):
    """验证文档是否已成功删除"""
    # 检查文档是否仍存在
    doc_status = await rag.doc_status.get_by_id(doc_id)
    if doc_status:
        print(f"警告: 文档 '{doc_id}' 仍存在于文档状态中")
        return False
    
    # 检查文档的文本块是否仍存在
    chunks = await rag.text_chunks.get_by_id(doc_id)
    if chunks:
        print(f"警告: 文档 '{doc_id}' 的文本块仍存在")
        return False
    
    # 检查原始文档是否仍存在
    full_doc = await rag.full_docs.get_by_id(doc_id)
    if full_doc:
        print(f"警告: 文档 '{doc_id}' 仍存在于全文档存储中")
        return False
    
    return True


async def async_delete_by_entity(entity_name):
    """异步方式根据实体名删除数据"""
    rag = initialize_rag()
    print(f"正在删除实体: {entity_name}")
    
    # 直接调用异步方法
    await rag.adelete_by_entity(entity_name)
    
    # 确保索引更新完成
    await rag._delete_by_entity_done()
    
    # 手动调用索引完成回调以确保数据库更新
    for storage_inst in [
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunk_entity_relation_graph,
    ]:
        if hasattr(storage_inst, "index_done_callback"):
            await storage_inst.index_done_callback()
    
    # 验证删除是否成功
    success = await verify_entity_deletion(rag, entity_name)
    if success:
        print(f"实体 '{entity_name}' 及其关系已被成功删除。")
    else:
        print(f"实体 '{entity_name}' 可能未完全删除，请检查。")


def delete_by_entity(entity_name):
    """根据实体名删除数据的同步包装器"""
    asyncio.run(async_delete_by_entity(entity_name))


async def async_delete_by_doc_id(doc_id):
    """异步方式根据文档ID删除数据"""
    rag = initialize_rag()
    print(f"正在删除文档: {doc_id}")
    
    # 调用异步删除方法
    await rag.adelete_by_doc_id(doc_id)
    
    # 确保索引更新完成
    await rag._insert_done()
    
    # 手动调用索引完成回调以确保数据库更新
    for storage_inst in [
        rag.full_docs,
        rag.doc_status,
        rag.text_chunks,
        rag.chunks_vdb,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunk_entity_relation_graph,
    ]:
        if hasattr(storage_inst, "index_done_callback"):
            await storage_inst.index_done_callback()
    
    # 验证删除是否成功
    success = await verify_doc_deletion(rag, doc_id)
    if success:
        print(f"文档 '{doc_id}' 及其相关数据已被成功删除。")
    else:
        print(f"文档 '{doc_id}' 可能未完全删除，请检查。")


def delete_by_doc_id(doc_id):
    """根据文档ID删除数据的同步包装器"""
    asyncio.run(async_delete_by_doc_id(doc_id))


def list_entities():
    """列出所有实体"""
    rag = initialize_rag()
    try:
        # 获取图中的所有节点
        entities = list(rag.chunk_entity_relation_graph._graph.nodes())
        if entities:
            print("已找到以下实体:")
            for entity in entities:
                print(f"  - {entity}")
        else:
            print("未找到任何实体。")
    except Exception as e:
        print(f"列出实体时出错: {e}")


def list_documents():
    """列出所有文档"""
    rag = initialize_rag()
    try:
        # 尝试获取所有文档ID
        # 注意：这里的实现可能需要根据LightRAG的实际API调整
        try:
            # 首先尝试通过client_storage获取
            docs = rag.full_docs.client_storage.get("data", [])
            doc_ids = [doc.get("id", "未知ID") for doc in docs if "id" in doc]
        except:
            # 如果上面的方法不可用，尝试使用其他方法
            # 这是一个备选方案，可能需要根据实际情况调整
            doc_ids = []
            print("无法获取文档列表，请检查LightRAG API或联系开发者。")
            
        if doc_ids:
            print("已找到以下文档:")
            for doc_id in doc_ids:
                print(f"  - {doc_id}")
        else:
            print("未找到任何文档。")
    except Exception as e:
        print(f"列出文档时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LightRAG删除工具 - 根据实体名或文档ID删除数据")
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 删除实体命令
    entity_parser = subparsers.add_parser("entity", help="根据实体名删除数据")
    entity_parser.add_argument("name", help="要删除的实体名")
    
    # 删除文档命令
    doc_parser = subparsers.add_parser("document", help="根据文档ID删除数据")
    doc_parser.add_argument("id", help="要删除的文档ID")
    
    # 列出实体命令
    list_entities_parser = subparsers.add_parser("list-entities", help="列出所有实体")
    
    # 列出文档命令
    list_docs_parser = subparsers.add_parser("list-documents", help="列出所有文档")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据命令执行相应的操作
    if args.command == "entity":
        delete_by_entity(args.name)
    elif args.command == "document":
        delete_by_doc_id(args.id)
    elif args.command == "list-entities":
        list_entities()
    elif args.command == "list-documents":
        list_documents()
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 