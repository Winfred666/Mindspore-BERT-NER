from wxauto import *

wx = WeChat()  # 获取当前微信客户端
wx.GetSessionList()  # 获取会话列表

def get_default_messages():
    # 调用wxauto中的方法：GetAllMessage
    msgs = wx.GetAllMessage()
    for msg in msgs:
        print('%s : %s' % (msg[0], msg[1]))
        
if __name__ == '__main__':
	get_default_messages()
