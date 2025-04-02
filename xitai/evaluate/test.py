from zhipuai import ZhipuAI
client = ZhipuAI(api_key="832e3f1798e5428ab763e2e109e193bf.53ME4IVuBzst0W4X")  # 请填写您自己的APIKey、

def generate_evaluation_prompt(question, standard_answer, answer):
    return f"""请评估以下回答的质量。标准答案和待评估答案如下：

    问题：
    {question}

    标准答案：
    {standard_answer}

    待评估答案：
    {answer}

    请根据以下标准评估回答：
    1. 是否包含或表达了标准答案中的关键信息,是否能认为待评估答案回答正确
    2. 是否有明显的错误性信息

    请只返回数字1（表示答案正确）或0（表示答案不正确），不要包含任何其他文字。"""

question = "修改ECN变更号的菜单路径是什么？"
standard_answer = "路径：SAP菜单 -> 交叉应用组件 ->工程变更管理-> 变更编号-> CC02-修改（事务代码：CC02 ） "
answer = "修改ECN变更号的菜单路径为：SAP菜单 -> 交叉应用组件 -> 工程变更管理 -> 变更编号 -> CC02-修改（事务代码：CC02）。"


response = client.chat.completions.create(
    model="glm-4-plus",  # 请填写您要调用的模型名称
    messages=[
        {"role": "user", "content": generate_evaluation_prompt(question, standard_answer, answer)},
    ],
    stream=False
)
print(response.choices[0].message.content)