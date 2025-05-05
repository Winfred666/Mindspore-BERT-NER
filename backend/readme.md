# 借助wxauto项目的可以通过电脑微信界面获取对话消息的程序

<!-- one_test_ner_backend.py
实现了从前端获取格式化文本，分解出其中的元素并标记位置

one_test_quantiDCE_backend.py
实现了从前端获取两段文本，比较之间的关联性并返回

one_test_ner_txt_backend.py
实现了从已有的对话文件（如：肖一鸣_chat_history.txt）中提取出对话，并对其进行ner文本元素识别，将得到的识别结果放在当前目录下的ner_results.json文件中

getbywindows_backend.py
实现了指定好友名字，获取跟其最近的对话内容并保存到txt文件中 -->


## wechat_manychats_to_json_adjust.py
实现了从微信获取聊天消息，并输出到user_chat文件夹中

/add_chat
该端口实现将指定名称的对话窗口加入列表中，并将列表存在specified_chats.json中

发送的body：
```
{
  "name": "周宇"
}
```
/get_all_chats
该端口实现从specified_chats.json得到所有被选中的对话窗口，获取对话窗口的消息并更新到user_chat中

返回的json格式为：
```
[
    {
        "id": 1,
        "sender": "Self",
        "message": "[图片]",
        "entities": [],
        "related_messages": []
    },
    {
        "id": 2,
        "sender": "肖一鸣",
        "message": "整挺好",
        "entities": [],
        "related_messages": []
    },
    {
        "id": 3,
        "sender": "Self",
        "message": "[动画表情]",
        "entities": [],
        "related_messages": []
    },
    {
        "id": 4,
        "sender": "SYS",
        "message": "2025年05月02日 16:57",
        "entities": [],
        "related_messages": []
    }
]
```
## send_file_to_server 中的 send_json.py 
该程序实现了将本地的json发送给服务器，以便进行推理处理，并含有向服务器拉取已经处理好的json的功能

## one_test_ner_json.py 
该程序部署在后端，实现了将已有的json文件中的实体识别出来并保存回json文件


# 后端进行推理判别的程序 在server_process文件夹内

## one_test_quantiDCE_json.py
实现将json内有已有消息跟其之前的100条范围内的消息进行比对，最多找5条score大于0.7的消息进行标记

## one_test_ner_json.py
提取已有json消息中的实体，并标记位置