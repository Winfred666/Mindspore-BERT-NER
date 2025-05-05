from wxauto import *
import time

wx = WeChat()
wx.GetSessionList()

def get_more_messages(name):
    try:
        wx.ChatWith(name)
        # 加载更多历史消息
        success = wx.LoadMoreMessage()
        if success:
            msgs = wx.GetAllMessage()
            for msg in msgs:
                print('%s : %s' % (msg[0], msg[1]))
        else:
            print("加载更多消息失败")
    except Exception as e:
        print(f"获取消息时出错: {e}")

if __name__ == '__main__':
    name = "肖一鸣"
    get_more_messages(name)