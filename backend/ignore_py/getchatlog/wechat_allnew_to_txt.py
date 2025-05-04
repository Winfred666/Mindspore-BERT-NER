from wxauto import *
import time
from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)
wx = WeChat()
wx.GetSessionList()

# 指定保存 JSON 文件的文件夹
user_chat_dir = "user_chat"

def format_message(msg):
    """格式化消息，处理换行符"""
    formatted_msg = msg.replace('\n', '[换行]')
    return formatted_msg

def load_existing_chat_data(file_path):
    """加载已有的聊天数据"""
    if not os.path.exists(file_path):
        return None, 0
    with open(file_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    max_id = max(item['id'] for item in existing_data) if existing_data else 0
    return existing_data, max_id

@app.route('/get_all_chats', methods=['POST'])
def get_all_chats():
    wx = WeChat()
    wx.GetSessionList()
    try:
        new_messages = wx.GetAllNewMessage()
        if new_messages:
            for chat_name, msgs in new_messages.items():
                print(f"新消息来自：{chat_name}")
                for msg in msgs:
                    print(f"{msg[0]} : {msg[1]}")
        else:
            print("没有新消息")
    except Exception as e:
        print(f"获取新消息时出错: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)