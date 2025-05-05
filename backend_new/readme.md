# 借助wxauto项目的可以通过电脑微信界面获取对话消息的程序

<!-- one_test_ner_backend.py
实现了从前端获取格式化文本，分解出其中的元素并标记位置

one_test_quantiDCE_backend.py
实现了从前端获取两段文本，比较之间的关联性并返回

one_test_ner_txt_backend.py
实现了从已有的对话文件（如：肖一鸣_chat_history.txt）中提取出对话，并对其进行ner文本元素识别，将得到的识别结果放在当前目录下的ner_results.json文件中

getbywindows_backend.py
实现了指定好友名字，获取跟其最近的对话内容并保存到txt文件中 -->

<!-- 
## wechat_manychats_to_json_adjust.py
实现了从微信获取聊天消息，并输出到user_chat文件夹中 -->

后端程序：one_total.py  

## POST /add_chat  
该端口实现将指定名称的对话窗口加入列表中，并将列表存在specified_chats.json中  
发送body：
```
{
  "name": "张三"
}
```
返回json消息：  
```
{
    "result": "Chat '周宇' added successfully"
}
```
## POST /delete_chat  
实现将指定名称的对话窗口从specified_chats.json的列表中移除  
发送body：  
```
{
  "name": "张三"
}
```
## POST /get_all_chats  
该端口实现从specified_chats.json得到所有被选中的对话窗口，获取对话窗口的消息并更新到user_chat中  
返回的json：
```
{
    "result": "All chats processed successfully"
}
```
## POST /get_all_chats
将specified_chats.json列表中所有名称对应的对话窗口更新并存储下来到user_chat文件夹
保存的json文件：
```
[
        {
        "id": 20,
        "sender": "Self",
        "message": "不过应该不影响",
        "entities": [],
        "is_quotation": false,
        "related_messages": [],
        "referenced_message": null
    },
    {
        "id": 21,
        "sender": "SYS",
        "message": "2025年05月05日 9:18",
        "entities": [],
        "is_quotation": false,
        "related_messages": [],
        "referenced_message": null
    },
    {
        "id": 22,
        "sender": "张三",
        "message": "在这个位置",
        "entities": [],
        "is_quotation": true,
        "related_messages": [
            {
                "id": 8,
                "score": 1.0
            }
        ],
        "referenced_message": "它的文件是哪一个"
    }
]
```
## POST /delete_file
根据specified_chats.json列表的更新删除已经从列表中被移除的名称对应的对话文件，和/get_all_chats 相反
返回json消息：
```
{
    "result": "Files for 肖一鸣 deleted successfully"
}
```
## GET /view_chats
将specified_chats.json中的列表输出  
返回json消息：
```
{
    "chats": [
        "张三",
        "李四"
    ]
}
```

<!-- ## send_file_to_server 中的 send_json.py 
该程序实现了将本地的json发送给服务器，以便进行推理处理，并含有向服务器拉取已经处理好的json的功能

## one_test_ner_json.py 
该程序部署在后端，实现了将已有的json文件中的实体识别出来并保存回json文件


# 后端进行推理判别的程序 在server_process文件夹内

## one_test_quantiDCE_json.py
实现将json内有已有消息跟其之前的100条范围内的消息进行比对，最多找5条score大于0.7的消息进行标记

## one_test_ner_json.py
提取已有json消息中的实体，并标记位置 -->

## POST /perform_entity_recognition
对已有的对话文件json进行实体识别，返回并返回标签和对应位置，保存回json文件  
返回json消息：
```
{
    "result": "Entity recognition completed successfully"
}
```
保存后的json文件：
```
[
    {
        "id": 21,
        "sender": "SYS",
        "message": "2025年05月05日 9:18",
        "entities": [
            {
                "entity": "Time",
                "range": [
                    0,
                    15
                ]
            }
        ],
        "is_quotation": false,
        "related_messages": [],
        "referenced_message": null
    }
]
```


## POST /analyze_relationships
对已有的对话文件json进行文本关联性分析，将json内有已有消息跟其之前的100条范围内的消息进行比对，最多找5条score大于0.5的消息进行标记，返回标签和对应位置，保存回json文件，通过引用找到的关联性文件不经过模型推理，score的值直接设为1  
返回json消息：
```
{
    "result": "Message relationships analyzed successfully"
}
```
保存回的json文件：
```
[
    {
        "id": 20,
        "sender": "Self",
        "message": "不过应该不影响部署后",
        "entities": [
            {
                "entity": "Time",
                "range": [
                    9,
                    10
                ]
            }
        ],
        "is_quotation": false,
        "related_messages": [
            {
                "id": 15,
                "score": 0.6482419967651367
            },
            {
                "id": 18,
                "score": 0.5906756520271301
            },
            {
                "id": 19,
                "score": 0.5752671360969543
            },
            {
                "id": 5,
                "score": 0.5286100506782532
            }
        ],
        "referenced_message": null
    }
]
```

