import os
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv()

# 初始化 LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatDeepSeek(model="deepseek-chat",
                   api_key=os.getenv("DEEPSEEK_API_KEY"),
                   temperature=0.5,
                   )

messages = [
    'hello',
    'world',
]

response = llm.invoke(messages)
print(response)
