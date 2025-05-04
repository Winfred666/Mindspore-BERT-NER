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
        file_path = f"{chat_name}_chat_history.txt"
        # 检查文件是否存在
        file_exists = os.path.exists(file_path)
        # 打开文件，准备写入。如果文件存在，追加模式；否则，写入模式。
        with open(file_path, 'a' if file_exists else 'w', encoding='utf-8') as f:
            # 如果文件存在，添加一个标记，表示从这里开始是新追加的内容
            if file_exists:
                f.write("\n" + "="*20 + " New Messages Start " + "="*20 + "\n")
            for msg in msgs:
                formatted_msg = msg[1].replace('\n', '[换行]')
                f.write('%s : %s\n' % (msg[0], formatted_msg))
        print(f"新消息已保存到 {file_path}")

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