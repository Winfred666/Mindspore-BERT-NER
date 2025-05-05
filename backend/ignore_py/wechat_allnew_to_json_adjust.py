from wxauto import *
import time
from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)
wx = WeChat()  # 在应用启动时初始化微信对象

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
    try:
        # 获取所有新消息
        wx.GetSessionList()  # 确保会话列表是最新的
        new_messages = wx.GetAllNewMessage()
        if not new_messages:
            return jsonify({"result": "No new messages"}), 200

        # 遍历每个对话窗口的消息
        for chat_name, msgs in new_messages.items():
            # 准备文件路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            user_chat_path = os.path.join(current_dir, user_chat_dir)
            if not os.path.exists(user_chat_path):
                os.makedirs(user_chat_path)
            file_path = os.path.join(user_chat_path, f"{chat_name}_chat_results.json")

            # 加载现有数据
            existing_data, max_id = load_existing_chat_data(file_path)

            # 准备结果数据
            results = existing_data if existing_data else []
            start_id = max_id + 1 if existing_data else 1

            # 处理新消息并追加到结果中
            for idx, msg in enumerate(msgs, start=start_id):
                sender = msg[0]
                content = msg[1]
                formatted_content = format_message(content)
                results.append({
                    "id": idx,
                    "sender": sender,
                    "message": formatted_content,
                    "entities": []
                })

            # 保存结果到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        return jsonify({"result": "All chats processed successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing chats: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)