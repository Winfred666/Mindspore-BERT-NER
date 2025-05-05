from wxauto import *
import time
from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)
wx = WeChat()  # 在应用启动时初始化微信对象

# 指定保存 JSON 文件的文件夹
user_chat_dir = "user_chat"
specified_chats_file = "specified_chats.json"  # 保存指定对话窗口名称的 JSON 文件

def format_message(msg):
    """格式化消息，处理换行符"""
    formatted_msg = msg.replace('\n', '[换行]')
    return formatted_msg

def load_specified_chats():
    """加载已指定的对话窗口名称"""
    if not os.path.exists(specified_chats_file):
        return []
    with open(specified_chats_file, 'r', encoding='utf-8') as f:
        specified_chats = json.load(f)
    return specified_chats

def save_specified_chats(specified_chats):
    """保存指定的对话窗口名称"""
    with open(specified_chats_file, 'w', encoding='utf-8') as f:
        json.dump(specified_chats, f, ensure_ascii=False, indent=4)

def load_chat_data(file_path):
    """加载聊天数据"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)
    return chat_data

def save_chat_data(file_path, chat_data):
    """保存聊天数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=4)

@app.route('/add_chat', methods=['POST'])
def add_chat():
    """添加指定的对话窗口名称"""
    data = request.json
    if not data or 'name' not in data:
        return jsonify({"error": "Invalid input format"}), 400
    name = data['name']
    
    specified_chats = load_specified_chats()
    if name not in specified_chats:
        specified_chats.append(name)
        save_specified_chats(specified_chats)
    
    return jsonify({"result": f"Chat '{name}' added successfully"})

@app.route('/get_all_chats', methods=['POST'])
def get_all_chats():
    """获取所有指定对话窗口的消息"""
    try:
        # 加载已指定的对话窗口名称
        specified_chats = load_specified_chats()
        if not specified_chats:
            return jsonify({"result": "No specified chats"}), 200
        
        # 遍历每个指定的对话窗口
        for name in specified_chats:
            # 确保微信客户端有该聊天窗口
            if not wx.ChatWith(name):
                print(f"Chat window '{name}' not found")
                continue
            
            # 获取所有消息
            msgs = wx.GetAllMessage(
                savepic=False,  # 保存图片
                savefile=False,  # 保存文件
                savevoice=True   # 保存语音转文字内容
            )
            
            # 准备文件路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            user_chat_path = os.path.join(current_dir, user_chat_dir)
            if not os.path.exists(user_chat_path):
                os.makedirs(user_chat_path)
            file_path = os.path.join(user_chat_path, f"{name}_chat_results.json")
            
            # 加载现有聊天数据
            chat_data = load_chat_data(file_path)
            
            # 处理新消息并追加到聊天数据中
            for msg in msgs:
                sender = msg[0]
                content = msg[1]
                formatted_content = format_message(content)
                chat_data.append({
                    "sender": sender,
                    "message": formatted_content,
                    "entities": []
                })
            
            # 保存聊天数据到文件
            save_chat_data(file_path, chat_data)
        
        return jsonify({"result": "All chats processed successfully"})
    
    except Exception as e:
        return jsonify({"error": f"Error processing chats: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)