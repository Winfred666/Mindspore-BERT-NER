from wxauto import WeChat

wx = WeChat()

# 获取当前聊天窗口消息
msgs = wx.GetAllMessage()

# 输出消息内容
for msg in msgs:
    if msg.type == 'sys':
        print(f'【系统消息】{msg.content}')
    
    elif msg.type == 'friend':
        sender = msg.sender_remark # 这里可以将msg.sender改为msg.sender_remark，获取备注名
        print(f'{sender.ljust(20)}：{msg.content}')

    elif msg.type == 'self':
        print(f'{msg.sender.ljust(20)}：{msg.content}')
    
    elif msg.type == 'time':
        print(f'\n【时间消息】{msg.time}')

    elif msg.type == 'recall':
        print(f'【撤回消息】{msg.content}')