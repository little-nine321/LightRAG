import json

# 打开并读取 JSON 文件
with open('./xitai/ragtest/kv_store_llm_response_cache.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print(data)
