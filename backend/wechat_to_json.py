from wxauto import *
import time
from flask import Flask, request, jsonify
import os
import json
import re

app = Flask(__name__)
wx = WeChat()
wx.GetSessionList()

# 指定保存 JSON 文件的文件夹
user_chat_dir = "user_chat"

def format_message(msg):
    """格式化消息，处理换行符"""
    formatted_msg = msg.replace('\n', '[换行]')
    return formatted_msg

def get_more_messages(name):
    try:
        wx.ChatWith(name)
        success = wx.LoadMoreMessage()
        if success:
            msgs = wx.GetAllMessage(
                savepic=False,  # 保存图片
                savefile=False,  # 保存文件
                savevoice=True   # 保存语音转文字内容
            )
            return msgs
        else:
            return None
    except Exception as e:
        print(f"获取消息时出错: {e}")
        return None

def load_existing_chat_data(file_path):
    """加载已有的聊天数据"""
    if not os.path.exists(file_path):
        return None, 0
    with open(file_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    max_id = max(item['id'] for item in existing_data) if existing_data else 0
    return existing_data, max_id

@app.route('/get_chat', methods=['POST'])
def get_chat():
    data = request.json
    if not data or 'name' not in data:
        return jsonify({"error": "Invalid input format"}), 400
    name = data['name']
    msgs = get_more_messages(name)
    if msgs is None:
        return jsonify({"error": "Failed to retrieve messages"}), 500
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    user_chat_path = os.path.join(current_dir, user_chat_dir)
    if not os.path.exists(user_chat_path):
        os.makedirs(user_chat_path)
    
    output_file_path = os.path.join(user_chat_path, f"{name}_chat_results.json")
    
    existing_data, max_id = load_existing_chat_data(output_file_path)
    
    results = existing_data if existing_data else []
    start_id = max_id + 1 if existing_data else 1
    
    new_entries = []
    for idx, msg in enumerate(msgs, start=start_id):
        sender = msg[0]
        content = msg[1]
        formatted_content = format_message(content)
        new_entries.append({
            "id": idx,
            "sender": sender,
            "message": formatted_content,
            "entities": []  
        })
    
    results.extend(new_entries)
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {e}"}), 500
    
    return jsonify({
        "results": results,
        "file_path": output_file_path
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)