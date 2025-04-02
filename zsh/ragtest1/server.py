from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from method import initialize_rag, query_rag, query_rag_stream, upload_document, get_document, delete_document, initialize_rag
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
from fastapi import File, UploadFile

# 问题1：默认只能上传txt
# 问题2：上传多个文件，部分文件出错，返回500报错，部分文件出错。

async def lifespan(app: FastAPI):
    global rag   # 声明 rag 为全局变量
    rag = await initialize_rag()
    yield

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    mode: str

@app.post("/query")
async def handle_query(request: QueryRequest):
    # 处理请求逻辑
    if not request.query or not request.mode:
        raise HTTPException(status_code=400, detail="Query and mode are required")

    if request.mode not in ["local", "global", "hybrid", "naive", "mix"]:
        raise HTTPException(status_code=400, detail="Mode must be 'local', 'global', 'hybrid', 'naive', or 'mix'")
    
    message = await query_rag(None, request.query, request.mode)

    return JSONResponse(content={"response": message})

@app.post("/query/stream")
async def handle_query_stream(request: QueryRequest):
    # 处理请求逻辑
    if not request.query or not request.mode:
        raise HTTPException(status_code=400, detail="Query and mode are required")

    if request.mode not in ["local", "global", "hybrid", "naive", "mix"]:
        raise HTTPException(status_code=400, detail="Mode must be 'local', 'global', 'hybrid', 'naive', or 'mix'")
    
    response = await query_rag_stream(None, request.query, request.mode)

    return StreamingResponse(
        response,
        media_type="text/event-stream"
    )
    
#上传文件api
@app.post("/document")
async def upload_documents(files: List[UploadFile] = File(...)):

    if not files:
        raise HTTPException(status_code=400, detail="请上传文件")
    
    flag = await upload_document(rag, files)
    if flag:
        return JSONResponse(content={"message": "上传成功"})
    else:
        return JSONResponse(content={"message": "存在文件上传失败"})

# 查看文件列表
@app.get("/document/list")
async def get_documents():
    files = await get_document()
    return JSONResponse(content={"response": files})

# 删除文件
@app.delete("/document")
async def delete_documents(file_id: str):

    flag = await delete_document(rag, file_id)
    if flag == 0:
        return JSONResponse(content={"message": "删除成功"})
    elif flag == 1:
        raise HTTPException(status_code=400, detail="文件不存在")
    else:
        raise HTTPException(status_code=500, detail="删除失败")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)