from wxauto import *

wx = WeChat()
wx.GetSessionList()


def get_single_messages(name):
    wx.ChatWith(name)
    msgs = wx.GetAllMessage()
    for msg in msgs:
        print('%s : %s' % (msg[0], msg[1]))


def get_multi_messages(names):
    for name in names:
        wx.ChatWith(name)
        msgs = wx.GetAllMessage()
        for msg in msgs:
            print('%s : %s' % (msg[0], msg[1]))

def ChatWith(self, who, RollTimes=None):
        '''
        打开某个聊天框
        who : 要打开的聊天框好友名，str;  * 最好完整匹配，不完全匹配只会选取搜索框第一个
        RollTimes : 默认向下滚动多少次，再进行搜索
        '''
        self.UiaAPI.SwitchToThisWindow()  
        RollTimes = 10 if not RollTimes else RollTimes
        # 当前显示的聊天列表中没找到指定名称的好友或群时，会滚动聊天列表界面，继续寻找
        def roll_to(who=who, RollTimes=RollTimes):
            for i in range(RollTimes):
                if who not in self.GetSessionList()[:-1]:
                    self.SessionList.WheelDown(wheelTimes=3, waitTime=0.1*i)
                else:
                    time.sleep(0.5)
                    # 这是点击客户端聊天列表中指定的窗口
                    self.SessionList.ListItemControl(Name=who).Click(simulateMove=False)
                    return 1
            return 0
        rollresult = roll_to()
        if rollresult:
            return 1
        else:
            self.Search(who)  # 当前显示的聊天列表中没找到指定名称的好友或群时，直接在搜索框中搜索
            return roll_to(RollTimes=1)



if __name__ == '__main__':
    name = "肖一鸣"
    get_single_messages(name)

    names = ['肖一鸣', 'HABITAT fans']
    get_multi_messages(names)
