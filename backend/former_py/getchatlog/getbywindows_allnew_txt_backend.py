from wxauto import *
import time
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
wx = WeChat()
wx.GetSessionList()

def save_new_messages(new_messages):
    """
    保存新消息到对应的文本文件中。
    如果文件已存在，则在文件末尾追加新消息；否则创建新文件。
    """
    for chat_name, msgs in new_messages.items():
        print(f"新消息来自：{chat_name}")
        for msg in msgs:
            print(f"{msg[0]} : {msg[1]}")

def get_and_save_new_messages():
    """
    获取所有新消息并保存到文件。
    """
    try:
        new_messages = wx.GetAllNewMessage()
        if new_messages:
            save_new_messages(new_messages)
            return "新消息已保存"
        else:
            return "没有新消息"
    except Exception as e:
        return f"获取新消息时出错: {e}"

@app.route('/get_new_messages', methods=['POST'])
def handle_get_new_messages():
    result = get_and_save_new_messages()
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)