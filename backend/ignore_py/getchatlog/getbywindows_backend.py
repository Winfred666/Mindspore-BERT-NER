# from wxauto import *
# import time
# from flask import Flask, request, jsonify
# import os

# app = Flask(__name__)
# wx = WeChat()
# wx.GetSessionList()

# def get_more_messages(name):
#     try:
#         wx.ChatWith(name)
#         # 加载更多历史消息
#         success = wx.LoadMoreMessage()
#         if success:
#             msgs = wx.GetAllMessage()
#             # 定义文件路径为当前路径
#             file_path = f"{name}_chat_history.txt"
#             # 确保当前目录存在
#             current_dir = os.path.dirname(os.path.abspath(__file__))
#             file_path = os.path.join(current_dir, file_path)
#             # 打开文件，准备写入
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 for msg in msgs:
#                     f.write('%s : %s\n' % (msg[0], msg[1]))
#             return f"聊天记录已保存到 {file_path}"
#         else:
#             return "加载更多消息失败"
#     except Exception as e:
#         return f"获取消息时出错: {e}"

# @app.route('/get_chat', methods=['POST'])
# def get_chat():
#     data = request.json
#     if not data or 'name' not in data:
#         return jsonify({"error": "Invalid input format"}), 400
#     name = data['name']
#     result = get_more_messages(name)
#     return jsonify({"result": result})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


from wxauto import *
import time
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
wx = WeChat()
wx.GetSessionList()

def format_message(msg):
    """格式化消息，处理换行符"""
    # 将换行符替换为'[换行]'
    formatted_msg = msg.replace('\n', '[换行]')
    return formatted_msg

def get_more_messages(name):
    try:
        wx.ChatWith(name)
        # 加载更多历史消息
        success = wx.LoadMoreMessage()
        if success:
            msgs = wx.GetAllMessage(
                savepic   = False,   # 保存图片
                savefile  = False,   # 保存文件
                savevoice = True    # 保存语音转文字内容
            )
            # 定义文件路径为当前路径
            file_path = f"{name}_chat_history.txt"
            # 确保当前目录存在
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, file_path)
            # 打开文件，准备写入
            with open(file_path, 'w', encoding='utf-8') as f:
                for msg in msgs:
                    # 格式化消息
                    formatted_msg = format_message(msg[1])
                    f.write('%s :\n%s\n' % (msg[0], formatted_msg))
            return f"聊天记录已保存到 {file_path}"
        else:
            return "加载更多消息失败"
    except Exception as e:
        return f"获取消息时出错: {e}"

@app.route('/get_chat', methods=['POST'])
def get_chat():
    data = request.json
    if not data or 'name' not in data:
        return jsonify({"error": "Invalid input format"}), 400
    name = data['name']
    result = get_more_messages(name)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)