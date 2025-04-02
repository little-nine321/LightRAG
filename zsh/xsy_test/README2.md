# LightRAG 删除工具

这个工具提供了一个简单的命令行界面，用于从 LightRAG 系统中删除实体和文档。

## 功能特点

- 根据实体名删除实体及其关系
- 根据文档ID删除文档及其相关数据（文本块、实体、关系）
- 列出所有实体
- 列出所有文档

## 安装要求

确保已安装以下依赖：

```bash
pip install lightrag numpy nest-asyncio python-dotenv
```

## 配置

工具使用以下环境变量，可以通过 `.env` 文件或系统环境变量设置：

- `RAG_DIR`: 工作目录，默认为 "index_default"
- `LLM_MODEL`: 使用的语言模型，默认为 "deepseek-r1:1.5b"
- `EMBEDDING_MODEL`: 使用的嵌入模型，默认为 "bge-m3:latest"
- `EMBEDDING_MAX_TOKEN_SIZE`: 嵌入的最大令牌大小，默认为 8192
- `BASE_URL`: API基础URL，默认为 "http://127.0.0.1:11434/v1"
- `API_KEY`: API密钥，默认为 "None"

## 使用方法

### 删除实体

```bash
python delete_tool.py entity "实体名称"
```

例如：

```bash
python delete_tool.py entity "Project Gutenberg"
```

### 删除文档

```bash
python delete_tool.py document "文档ID"
```

例如：

```bash
python delete_tool.py document "doc_12345"
```

### 列出所有实体

```bash
python delete_tool.py list-entities
```

### 列出所有文档

```bash
python delete_tool.py list-documents
```

### 查看帮助

```bash
python delete_tool.py --help
```

## 工作原理

1. **删除实体**：调用 LightRAG 的 `delete_by_entity` 方法，删除指定名称的实体及其所有关系。

2. **删除文档**：调用 LightRAG 的 `delete_by_doc_id` 方法，删除指定ID的文档及其所有相关数据，包括：
   - 文档本身
   - 文档的所有文本块
   - 与这些文本块相关的所有实体
   - 与这些文本块相关的所有关系

3. **列出实体**：访问 LightRAG 的知识图谱，列出所有实体节点。

4. **列出文档**：访问 LightRAG 的文档存储，列出所有文档ID。

## 注意事项

- 删除操作是不可逆的，请谨慎操作。
- 删除实体会同时删除与该实体相关的所有关系。
- 删除文档会同时删除与该文档相关的所有文本块、实体和关系。
- 如果实体或文档不存在，工具会给出相应的提示。

## 示例

### 删除实体示例

```bash
$ python delete_tool.py entity "Charles Dickens"
正在删除实体: Charles Dickens
实体 'Charles Dickens' 及其关系已被删除。
```

### 删除文档示例

```bash
$ python delete_tool.py document "doc_12345"
正在删除文档: doc_12345
文档 'doc_12345' 及其相关数据已被删除。
```

### 列出实体示例

```bash
$ python delete_tool.py list-entities
已找到以下实体:
  - "CHARLES DICKENS"
  - "EBENEZER SCROOGE"
  - "BOB CRATCHIT"
  - "TINY TIM"
  - "JACOB MARLEY"
```

### 列出文档示例

```bash
$ python delete_tool.py list-documents
已找到以下文档:
  - doc_12345
  - doc_67890
```

## 最新改进

最新版本的删除工具包含以下改进：

1. **异步删除操作**：
   - 使用异步方法进行实体和文档删除
   - 确保删除操作完成后数据库被正确更新

2. **验证功能**：
   - 删除操作后自动验证删除是否成功
   - 检查实体是否仍存在于关系图和向量数据库中
   - 检查文档是否仍存在于文档状态、文本块和全文档存储中

3. **详细反馈**：
   - 提供详细的操作结果反馈
   - 如果删除不完全，显示警告消息

## 常见问题解决

如果删除操作后提示"可能未完全删除"，可以尝试以下解决方法：

1. **重试删除操作**：
   ```bash
   python delete_tool.py entity "实体名称"
   ```
   或
   ```bash
   python delete_tool.py document "文档ID"
   ```

2. **检查数据库状态**：
   ```bash
   python delete_tool.py list-entities
   ```
   或
   ```bash
   python delete_tool.py list-documents
   ```

3. **手动强制更新索引**：
   这需要直接访问LightRAG的API，可能需要编写额外的脚本。 