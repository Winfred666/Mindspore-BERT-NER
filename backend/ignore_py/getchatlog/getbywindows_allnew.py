from wxauto import *

wx = WeChat()  # 获取当前微信客户端
wx.GetSessionList()  # 获取会话列表

def get_and_print_new_messages():
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
    get_and_print_new_messages()