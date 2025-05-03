from wxauto import *
import time
from flask import Flask, request, jsonify
import os
import json
import datetime
import re  # 导入re模块

app = Flask(__name__)
wx = WeChat()  # 在应用启动时初始化微信对象

# 指定保存 JSON 文件的文件夹
user_chat_dir = "user_chat"

# 指定 specified_chats.json 的路径为 user_chat 文件夹内
specified_chats_file = os.path.join(user_chat_dir, "specified_chats.json")

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
    # 确保 user_chat 目录存在
    if not os.path.exists(user_chat_dir):
        os.makedirs(user_chat_dir)
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

def format_time(time_str):
    """格式化时间字符串为标准格式"""
    try:
        # 尝试匹配并转换时间格式
        # 模式1：2025年4月4日 21:55
        if re.match(r'^\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}:\d{2}$', time_str):
            return time_str
        # 模式2：星期* \d{1,2}:\d{2}，包括星期天
        elif re.match(r'^星期[一二三四五六天日] \d{1,2}:\d{2}$', time_str):
            # 获取当前日期
            current_date = datetime.datetime.now()
            # 解析星期和时间
            weekday_str = re.match(r'^星期([一二三四五六天日])', time_str).group(1)
            time_part = re.search(r' (\d{1,2}:\d{2})$', time_str).group(1) if re.search(r' (\d{1,2}:\d{2})$', time_str) else "00:00"
            # 将星期转换为数字
            weekday_map = {'一': 0, '二': 1, '三': 2, '四': 3, '五': 4, '六': 5, '天': 6, '日': 6}
            weekday_num = weekday_map.get(weekday_str, 0)
            # 计算目标日期
            target_date = current_date + datetime.timedelta(days=(weekday_num - current_date.weekday()) % 7)
            # 组合目标日期和时间
            formatted_time = target_date.strftime(f'%Y年%m月%d日 {time_part}')
            return formatted_time
        # 模式3：昨天 \d{1,2}:\d{2}
        elif re.match(r'^昨天 \d{1,2}:\d{2}$', time_str):
            # 获取昨天的日期
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            time_part = re.search(r' (\d{1,2}:\d{2})$', time_str).group(1) if re.search(r' (\d{1,2}:\d{2})$', time_str) else "00:00"
            formatted_time = yesterday.strftime(f'%Y年%m月%d日 {time_part}')
            return formatted_time
        # 模式4：\d{1,2}:\d{2}
        elif re.match(r'^\d{1,2}:\d{2}$', time_str):
            # 获取当前日期
            current_date = datetime.datetime.now()
            # 组合当前日期和时间
            formatted_time = current_date.strftime(f'%Y年%m月%d日 {time_str}')
            return formatted_time
        else:
            return time_str
    except Exception as e:
        print(f"时间格式转换出错: {e}")
        return time_str

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
            
            # 比对新消息与现有消息的最后3条
            if len(chat_data) >= 3:
                existing_last_3 = [msg['message'] for msg in chat_data[-3:]]
                
                # 查找新消息中与现有最后3条相同的位置
                for i in range(len(msgs) - 2):
                    new_3 = [format_message(msg[1]) for msg in msgs[i:i+3]]
                    if new_3 == existing_last_3:
                        # 只保留新消息中从i+3开始的部分
                        msgs = msgs[i+3:]
                        break
            
            # 处理新消息并追加到聊天数据中
            new_chat_data = []
            for msg in msgs:
                sender = msg[0]
                content = msg[1]
                formatted_content = format_message(content)
                
                # 如果是系统消息，尝试提取并格式化时间
                if sender == 'SYS':
                    # 尝试匹配时间格式
                    time_match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}:\d{2})|'
                                           r'(星期[一二三四五六天日] \d{1,2}:\d{2})|'
                                           r'(昨天 \d{1,2}:\d{2})|'
                                           r'(\d{1,2}:\d{2})', formatted_content)
                    if time_match:
                        # 提取时间部分并格式化
                        time_str = time_match.group(0)
                        formatted_time = format_time(time_str)
                        
                        # 替换原消息中的时间部分
                        formatted_content = re.sub(r'(\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}:\d{2})|'
                                                   r'(星期[一二三四五六天日] \d{1,2}:\d{2})|'
                                                   r'(昨天 \d{1,2}:\d{2})|'
                                                   r'(\d{1,2}:\d{2})', formatted_time, formatted_content)
                
                new_chat_data.append({
                    "id": len(chat_data) + len(new_chat_data) + 1,  # 为每条新消息生成一个唯一的id
                    "sender": sender,
                    "message": formatted_content,
                    "entities": []
                })
            
            # 将新消息添加到聊天数据中
            chat_data.extend(new_chat_data)
            
            # 保存聊天数据到文件
            save_chat_data(file_path, chat_data)
        
        return jsonify({"result": "All chats processed successfully"})
    
    except Exception as e:
        return jsonify({"error": f"Error processing chats: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)