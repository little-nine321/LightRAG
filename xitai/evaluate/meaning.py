from zhipuai import ZhipuAI
import pandas as pd
import json

client = ZhipuAI(api_key="832e3f1798e5428ab763e2e109e193bf.53ME4IVuBzst0W4X")  # 请填写您自己的APIKey

def generate_evaluation_prompt(question, standard_answer, answer):
    return f"""请评估以下回答的质量。标准答案和待评估答案如下：

    问题：
    {question}

    标准答案：
    {standard_answer}

    待评估答案：
    {answer}

    请根据以下标准评估回答：
    1. 是否包含或表达了标准答案中的信息
    2. 是否有明显的错误性信息

    请只返回数字1（表示答案正确）或0（表示答案不正确），不要包含任何其他文字。"""

# 打开并读取 JSON 文件
with open('./xitai/evaluate/answer2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# now_index = data[0]["evaluate_index"]
# for index, item in enumerate(data):

#     if index < now_index:
#         continue
    
#     for method in ["local", "global", "naive", "hybrid", "mix"]:

#         response = client.chat.completions.create(
#             model="glm-4-plus", 
#             messages=[
#                 {"role": "user", "content": generate_evaluation_prompt(item["question"], item["true_answer"], item[method])}
#             ],
#             stream=False,
#         )

#         item[method + "_evaluate"] = response.choices[0].message.content

#     print(f"完成第{now_index}个问题")

#     now_index += 1
#     data[0]["evaluate_index"] = now_index

#     with open("./xitai/evaluate/answer2.json", "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

naive_total_time = 0
local_total_time = 0
global_total_time = 0
hybrid_total_time = 0
mix_total_time = 0

naive_total_evaluate = 0
local_total_evaluate = 0
global_total_evaluate = 0
hybrid_total_evaluate = 0
mix_total_evaluate = 0

for index, item in enumerate(data):

    if index == 0:
        continue

    naive_total_time += float(item["naive_time"].split("秒")[0])
    naive_total_evaluate += int(item["naive_evaluate"] == "1")

    local_total_time += float(item["local_time"].split("秒")[0])
    local_total_evaluate += int(item["local_evaluate"] == "1")

    global_total_time += float(item["global_time"].split("秒")[0])
    global_total_evaluate += int(item["global_evaluate"] == "1")   

    hybrid_total_time += float(item["hybrid_time"].split("秒")[0])
    hybrid_total_evaluate += int(item["hybrid_evaluate"] == "1")

    mix_total_time += float(item["mix_time"].split("秒")[0])
    mix_total_evaluate += int(item["mix_evaluate"] == "1")


data[0]["naive_avg_time"] = str(naive_total_time / 138)  + "秒"
data[0]["naive_avg_evaluate"] = naive_total_evaluate / 138

data[0]["local_avg_time"] = str(local_total_time / 138) + "秒"
data[0]["local_avg_evaluate"] = local_total_evaluate / 138

data[0]["global_avg_time"] = str(global_total_time / 138) + "秒"
data[0]["global_avg_evaluate"] = global_total_evaluate / 138

data[0]["hybrid_avg_time"] = str(hybrid_total_time / 138) + "秒"
data[0]["hybrid_avg_evaluate"] = hybrid_total_evaluate / 138

data[0]["mix_avg_time"] = str(mix_total_time / 138) + "秒"
data[0]["mix_avg_evaluate"] = mix_total_evaluate / 138

with open("./xitai/evaluate/answer2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
