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
## POST /update_all_chats
将specified_chats.json列表中所有名称对应的对话窗口更新并存储下来到user_chat文件夹
保存的单个json文件：
```
[
    {
        "entities": [],
        "id": 1,
        "is_quotation": false,
        "message": "web 形式",
        "referenced_message": null,
        "related_after": [],
        "related_before": [],
        "sender": "肖一鸣",
        "visible": true
    },
    {
        "entities": [],
        "id": 2,
        "is_quotation": false,
        "message": "嗯。。说不定不冲突？",
        "referenced_message": null,
        "related_after": [
            {
                "id": 4,
                "score": 1.0
            }
        ],
        "related_before": [],
        "sender": "Self",
        "visible": true
    }
]
```
返回的后端消息“
```
[
    {
        "chat_data": [
            (一个json文件的内容)
        ],
        "chat_name": "小明"
    },
    {
        "chat_data": [
            (第二个json的内容)
        ],
        "chat_name": "小红"
    }
]
```
## POST /get_chat_by_name
发送一个对话窗口名称，返回一个对应json文件里的所有内容  
发送消息：  
```
{
  "name": "小明"
}
```
返回消息：同/update_all_chats
## POST /update_message_visibility
发送名称、id和真假值，调整单段文本的可见性  
发送消息：
```
{
  "name": "小明",
  "id": 2,
  "visibility": false
}
```
返回消息同/get_chat_by_name
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
        "id": 14,
        "sender": "肖一鸣",
        "message": "那等你修好我再搞",
        "entities": [],
        "is_quotation": false,
        "related_before": [],
        "related_after": [],
        "referenced_message": null,
        "visible": true,
        "topic_start": false,
        "topic_quantity": 0,
        "topic_Tightness": 0.0
    },
    {
        "id": 15,
        "sender": "Self",
        "message": "其实再加个端口获取所有json就行",
        "entities": [],
        "is_quotation": false,
        "related_before": [
            {
                "id": 11,
                "score": 0.6478745937347412
            },
            {
                "id": 10,
                "score": 0.6008538603782654
            },
            {
                "id": 4,
                "score": 0.6010714769363403
            },
            {
                "id": 1,
                "score": 0.6109430193901062
            }
        ],
        "related_after": [],
        "referenced_message": null,
        "visible": true,
        "topic_start": false,
        "topic_quantity": 0,
        "topic_Tightness": 0.0
    }
]
```
## /analyze_topics
分析所有消息，对于有后索引而没有前索引的消息，将其判定为一个话题的开始，设置其topic_start为真，并且在topic一栏记录其中该消息通过后索引关联到的消息以及后索引的后索引等等以此类推关联到的所有消息的总数topic_quantity，并且计算话题的紧密度topic_Tightness，紧密度的计算方式如下：
假设判别关联性分数阈值为relatedscore，score(A,B)表示AB之间的关联性得分。那么一条分支链的紧密度为这条分支链开始端的score乘上后面所有分支的紧密度之和，到达链的最末端时紧密度为relatedscore。举例：假设目前话题的开始是A消息，只有A→B一条后索引链，那么topic_Tightness=score(A,B)*relatedscore。假设有A开始的topic有A→B，B→C，B→D，C→E，四条链，那么topic_Tightness=score(A,B)*(score(B,C)*score(C,E)*relatedscore+score(B,D)*relatedscore)
保存回的json：
```
[
    {
        "id": 1,
        "sender": "肖一鸣",
        "message": "web 形式",
        "entities": [],
        "is_quotation": false,
        "related_before": [],
        "related_after": [
            {
                "id": 5,
                "score": 0.7194541692733765
            },
            {
                "id": 11,
                "score": 0.6729263067245483
            },
            {
                "id": 6,
                "score": 0.6378445625305176
            },
            {
                "id": 8,
                "score": 0.6374056339263916
            },
            {
                "id": 10,
                "score": 0.6225969195365906
            },
            {
                "id": 15,
                "score": 0.6109430193901062
            }
        ],
        "referenced_message": null,
        "visible": true,
        "topic_start": true,
        "topic_quantity": 17,
        "topic_Tightness": 3.312966773010656
    }
]
```

## POST /clear_analyses
将/perform_entity_recognition、/analyze_relationships和/analyze_topics所写入json的信息全部清除，只保留获取到的原始消息文件（可见性的修改仍保留）


## POST /update_topic_visibility
指定一个消息窗口和id，如果这个消息是一个topic的开始，将这个topic下的所有消息的可见性进行改变
发送消息：
```
{
    "name": "小明",
    "id": 5,
    "visible": true
}
```

## POST /update_all_visible
指定一个聊天窗口，将其所有的消息的可见性都进行改变
```
{
    "name": "肖一鸣",
    "visible": true
}
```
