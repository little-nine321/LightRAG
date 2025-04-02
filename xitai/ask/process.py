import openpyxl
import json
import os

file_path = "./ask/熙泰测试数据集.xlsx"

# 打开xlsx文件
workbook = openpyxl.load_workbook(file_path)
sheet = workbook.active

# 创建一个列表来存储对象
data_list = []

# 读取第二列的值，从第二行开始记录
for row in sheet.iter_rows(min_row=2, min_col=2, max_col=2):
    for cell in row:
        data_list.append({
            "question": cell.value,
            "naive": "",
            "local": "",
            "global": "",
            "hybrid": "",
            "mix": ""
        })

# 获取文件路径的目录部分
dir_path = os.path.dirname(file_path)

# 定义保存的json文件路径
json_file_path = os.path.join(dir_path, "question.json")

# 将data_list保存为json格式文件
with open(json_file_path, "w", encoding="utf-8") as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)


