from wxauto import *
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import datetime
import re
import numpy as np
from src.dependency import convert_single_example, InputExample
import src.tokenization as tokenization
from transformers import AutoTokenizer
import onnx
import onnxruntime
import collections

app = Flask(__name__)
CORS(app)

# wx = WeChat()  # 在应用启动时初始化微信对象

max_retries = 3
retry_delay = 2  # 重试间隔，单位为秒

for _ in range(max_retries):
    try:
        wx = WeChat()
        break
    except pywintypes.error as e:
        print(f"Error initializing WeChat: {e}")
        time.sleep(retry_delay)
else:
    print("Failed to initialize WeChat after multiple retries.")

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 微信聊天管理部分
specified_chats_dir = os.path.join(current_dir, "friend_name_infor")
specified_chats_file = os.path.join(specified_chats_dir, "specified_chats.json")
user_chat_dir = os.path.join(current_dir, "user_chat")

def format_message(msg):
    """格式化消息，处理换行符以及引用消息"""
    formatted_msg = msg.replace('\n', '[换行]')
    
    # 检测并处理引用消息
    pattern = r'\[换行\]\s*引用\s+的消息\s*:\s*(.*)'
    quotation_match = re.search(pattern, formatted_msg)
    if quotation_match:
        # 提取引用部分
        referenced_message = quotation_match.group(1)
        # 删除引用部分
        formatted_msg = formatted_msg[:quotation_match.start()]
        # 标记为引用消息
        return formatted_msg, True, referenced_message
    
    return formatted_msg, False, None

def load_specified_chats():
    """加载已指定的对话窗口名称"""
    if not os.path.exists(specified_chats_file):
        return []
    with open(specified_chats_file, 'r', encoding='utf-8') as f:
        specified_chats = json.load(f)
    return specified_chats

def save_specified_chats(specified_chats):
    """保存指定的对话窗口名称"""
    # 确保 friend_name_infor 目录存在
    if not os.path.exists(specified_chats_dir):
        os.makedirs(specified_chats_dir)
    with open(specified_chats_file, 'w', encoding='utf-8') as f:
        json.dump(specified_chats, f, ensure_ascii=False, indent=4)

def load_chat_data(file_path):
    """加载聊天数据"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)
    return chat_data

def save_chat_data(file_path, chat_data):
    """保存聊天数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=4)

def format_time(time_str):
    """格式化时间字符串为标准格式"""
    try:
        # 尝试匹配并转换时间格式
        # 模式1：2025年4月4日 21:55
        if re.match(r'^\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}:\d{2}$', time_str):
            return time_str
        # 模式2：星期* \d{1,2}:\d{2}，包括星期天
        elif re.match(r'^星期[一二三四五六天日] \d{1,2}:\d{2}$', time_str):
            # 获取当前日期
            current_date = datetime.datetime.now()
            # 解析星期和时间
            weekday_str = re.match(r'^星期([一二三四五六天日])', time_str).group(1)
            time_part = re.search(r' (\d{1,2}:\d{2})$', time_str).group(1) if re.search(r' (\d{1,2}:\d{2})$', time_str) else "00:00"
            # 将星期转换为数字
            weekday_map = {'一': 0, '二': 1, '三': 2, '四': 3, '五': 4, '六': 5, '天': 6, '日': 6}
            weekday_num = weekday_map.get(weekday_str, 0)
            # 计算目标日期
            target_date = current_date + datetime.timedelta(days=(weekday_num - current_date.weekday()) % 7)
            # 组合目标日期和时间
            formatted_time = target_date.strftime(f'%Y年%m月%d日 {time_part}')
            return formatted_time
        # 模式3：昨天 \d{1,2}:\d{2}
        elif re.match(r'^昨天 \d{1,2}:\d{2}$', time_str):
            # 获取昨天的日期
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            time_part = re.search(r' (\d{1,2}:\d{2})$', time_str).group(1) if re.search(r' (\d{1,2}:\d{2})$', time_str) else "00:00"
            formatted_time = yesterday.strftime(f'%Y年%m月%d日 {time_part}')
            return formatted_time
        # 模式4：\d{1,2}:\d{2}
        elif re.match(r'^\d{1,2}:\d{2}$', time_str):
            # 获取当前日期
            current_date = datetime.datetime.now()
            # 组合当前日期和时间
            formatted_time = current_date.strftime(f'%Y年%m月%d日 {time_str}')
            return formatted_time
        else:
            return time_str
    except Exception as e:
        print(f"时间格式转换出错: {e}")
        return time_str

@app.route('/add_chat', methods=['POST'])
def add_chat():
    """添加指定的对话窗口名称"""
    data = request.json
    if not data or 'name' not in data:
        return jsonify({"error": "Invalid input format"}), 400
    name = data['name']
    
    specified_chats = load_specified_chats()
    if name not in specified_chats:
        specified_chats.append(name)
        save_specified_chats(specified_chats)
    
    return jsonify({"result": f"Chat '{name}' added successfully"})

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    """从指定对话窗口列表中移除指定名称的聊天窗口"""
    data = request.json
    if not data or 'name' not in data:
        return jsonify({"error": "Invalid input format"}), 400
    name = data['name']
    
    specified_chats = load_specified_chats()
    if name in specified_chats:
        specified_chats.remove(name)
        save_specified_chats(specified_chats)
    
    return jsonify({"result": f"Chat '{name}' deleted successfully"})


@app.route('/view_chats', methods=['GET'])
def view_chats():
    """查看指定对话窗口列表中现有的对话窗口名称"""
    specified_chats = load_specified_chats()
    if not specified_chats:
        return jsonify({"result": "No specified chats"}), 200
    return jsonify({"chats": specified_chats})

@app.route('/delete_file', methods=['POST'])
def delete_file():
    """删除user_chat中不在specified_chats.json列表中的对话文件"""
    # 加载指定的对话窗口名称列表
    specified_chats = load_specified_chats()

    # 获取user_chat目录路径
    user_chat_path = os.path.join(current_dir, user_chat_dir.lstrip("/"))
    if not os.path.exists(user_chat_path):
        return jsonify({"result": "user_chat directory not found"}), 404

    # 遍历user_chat目录中的所有文件
    deleted_files = []
    for file_name in os.listdir(user_chat_path):
        # 检查文件是否为_chat_results.json格式
        if file_name.endswith("_chat_results.json"):
            # 提取文件名中的名称部分
            name = file_name[:-18]  # 去掉"_chat_results.json"部分
            
            # 检查名称是否在指定的对话窗口列表中
            if name not in specified_chats:
                # 构建文件路径
                file_path = os.path.join(user_chat_path, file_name)
                # 删除文件
                os.remove(file_path)
                deleted_files.append(name)

    if deleted_files:
        return jsonify({"result": f"Files for {', '.join(deleted_files)} deleted successfully"})
    else:
        return jsonify({"result": "No files deleted. All chat files are up to date with specified chats list"})

@app.route('/update_all_chats', methods=['POST'])
def update_all_chats():
    """获取所有指定对话窗口的消息并更新引用关系"""
    try:
        # 加载已指定的对话窗口名称
        specified_chats = load_specified_chats()
        if not specified_chats:
            return jsonify({"result": "No specified chats"}), 200
        
        # 遍历每个指定的对话窗口
        all_chat_data = []  # 用于收集所有聊天数据
        for name in specified_chats:
            # 确保微信客户端有该聊天窗口
            if not wx.ChatWith(name):
                print(f"Chat window '{name}' not found")
                continue
            
            # 获取所有消息
            msgs = wx.GetAllMessage(
                savepic=False,  # 保存图片
                savefile=False,  # 保存文件
                savevoice=True   # 保存语音转文字内容
            )
            
            # 准备文件路径
            user_chat_path = os.path.join(current_dir, user_chat_dir.lstrip("/"))  # 确保路径正确
            if not os.path.exists(user_chat_path):
                os.makedirs(user_chat_path)
            file_path = os.path.join(user_chat_path, f"{name}_chat_results.json")
            
            # 加载现有聊天数据
            chat_data = load_chat_data(file_path)
            
            # 比对新消息与现有消息的最后3条
            if len(chat_data) >= 3:
                existing_last_3 = [msg['message'] for msg in chat_data[-3:]]
                
                # 查找新消息中与现有最后3条相同的位置
                for i in range(len(msgs) - 2):
                    new_3 = [format_message(msg[1])[0] for msg in msgs[i:i+3]]
                    if new_3 == existing_last_3:
                        # 只保留新消息中从i+3开始的部分
                        msgs = msgs[i+3:]
                        break
            
            # 处理新消息并追加到聊天数据中
            new_chat_data = []
            for msg in msgs:
                sender = msg[0]
                content = msg[1]
                formatted_content, is_quotation, referenced_message = format_message(content)
                
                # 如果是系统消息，尝试提取并格式化时间
                if sender == 'SYS':
                    # 尝试匹配时间格式
                    time_match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}:\d{2})|'
                                           r'(星期[一二三四五六天日] \d{1,2}:\d{2})|'
                                           r'(昨天 \d{1,2}:\d{2})|'
                                           r'(\d{1,2}:\d{2})', formatted_content)
                    if time_match:
                        # 提取时间部分并格式化
                        time_str = time_match.group(0)
                        formatted_time = format_time(time_str)
                        
                        # 替换原消息中的时间部分
                        formatted_content = re.sub(r'(\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}:\d{2})|'
                                                   r'(星期[一二三四五六天日] \d{1,2}:\d{2})|'
                                                   r'(昨天 \d{1,2}:\d{2})|'
                                                   r'(\d{1,2}:\d{2})', formatted_time, formatted_content)
                
                new_chat_data.append({
                    "id": len(chat_data) + len(new_chat_data) + 1,  # 为每条新消息生成一个唯一的id
                    "sender": sender,
                    "message": formatted_content,
                    "entities": [],
                    "is_quotation": is_quotation,  # 添加引用标记
                    "related_before": [],          # 添加 related_before 字段
                    "related_after": [],           # 添加 related_after 字段
                    "referenced_message": referenced_message if is_quotation else None,
                    "visible": True                # 新增可见性属性，默认为True
                })
            
            # 将新消息添加到聊天数据中
            chat_data.extend(new_chat_data)
            
            # 处理引用消息的 related_before 和 related_after
            for i, msg in enumerate(chat_data):
                if msg.get("is_quotation", False):
                    referenced_message = msg.get("referenced_message", "")
                    if referenced_message:
                        # 查找引用的消息（related_before）
                        for j in range(i):
                            if chat_data[j].get("message") == referenced_message:
                                msg["related_before"].append({
                                    "id": chat_data[j]["id"],
                                    "score": 1.0
                                })
                                # 同时更新被引用消息的 related_after
                                if "related_after" in chat_data[j]:
                                    chat_data[j]["related_after"].append({
                                        "id": msg["id"],
                                        "score": 1.0
                                    })
                                else:
                                    chat_data[j]["related_after"] = [{
                                        "id": msg["id"],
                                        "score": 1.0
                                    }]
                                break
            
            # 保存聊天数据到文件
            save_chat_data(file_path, chat_data)
            
            # 将当前聊天窗口的数据添加到 all_chat_data
            all_chat_data.append({
                "chat_name": name,
                "chat_data": chat_data
            })
        
        # 返回所有聊天数据作为 HTTP 响应
        return jsonify(all_chat_data), 200
    
    except Exception as e:
        return jsonify({"error": f"Error processing chats: {str(e)}"}), 500


@app.route('/update_message_visibility', methods=['POST'])
def update_message_visibility():
    """更新指定消息的可见性"""
    try:
        data = request.json
        if not data or 'name' not in data or 'id' not in data or 'visibility' not in data:
            return jsonify({"error": "Invalid input format"}), 400
        
        name = data['name']
        msg_id = data['id']
        visibility = data['visibility']
        
        if not isinstance(visibility, bool):
            return jsonify({"error": "Invalid visibility value. Must be true or false."}), 400
        
        user_chat_path = os.path.join(current_dir, user_chat_dir.lstrip("/"))
        if not os.path.exists(user_chat_path):
            return jsonify({"result": f"No chat found for {name}"}), 404
        
        file_path = os.path.join(user_chat_path, f"{name}_chat_results.json")
        if not os.path.exists(file_path):
            return jsonify({"result": f"No chat found for {name}"}), 404
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # 更新指定消息的可见性
        updated = False
        for msg in chat_data:
            if msg.get("id") == msg_id:
                msg["visible"] = visibility
                updated = True
                break
        
        if not updated:
            return jsonify({"result": f"Message with id {msg_id} not found"}), 404
        
        # 保存更新后的聊天数据
        save_chat_data(file_path, chat_data)
        
        # 返回更新后的所有聊天数据
        with open(file_path, 'r', encoding='utf-8') as f:
            updated_chat_data = json.load(f)
        
        return jsonify(updated_chat_data), 200
    
    except Exception as e:
        return jsonify({"error": f"Error updating message visibility: {str(e)}"}), 500
    
@app.route('/get_chat_by_name', methods=['POST'])
def get_chat_by_name():
    """根据对话窗口名称获取聊天记录"""
    try:
        data = request.json
        if not data or 'name' not in data:
            return jsonify({"error": "Invalid input format"}), 400
        
        name = data['name']
        user_chat_path = os.path.join(current_dir, user_chat_dir.lstrip("/"))
        if not os.path.exists(user_chat_path):
            return jsonify({"result": f"No chat found for {name}"}), 404
        
        file_path = os.path.join(user_chat_path, f"{name}_chat_results.json")
        if not os.path.exists(file_path):
            return jsonify({"result": f"No chat found for {name}"}), 404
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        return jsonify(chat_data), 200
    
    except Exception as e:
        return jsonify({"error": f"Error retrieving chat: {str(e)}"}), 500
    


# 实体识别部分
dataneeded_dir = os.path.join(current_dir, 'dataneeded')
vocab_file_ner = os.path.join(dataneeded_dir, 'vocab.txt')
onnx_path_ner = os.path.join(dataneeded_dir, 'chineseNER.onnx')
label_list_file_ner = os.path.join(dataneeded_dir, 'label_list.txt')

# use BMES representation, however can use other to do it.
def convert_labels_to_index(label_list):
    """
    Convert label_list to indices for NER task.
    """
    label2id = collections.OrderedDict()
    label2id["O"] = 0
    index = 0
    for label in label_list:
        if label == "O":
            continue
        index += 1
        sub_label = label
        label2id[sub_label] = index
    return label2id


# 实体识别相关配置
user_chat_dir = "user_chat"

label_list_ner = []
onnx_session_ner = None
tokenizer_ner = None
tag_to_index_ner = {}
max_seq_len_ner = 128

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 相对路径到 dataneeded 目录
dataneeded_dir = os.path.join(current_dir, 'dataneeded')

# 构建文件的相对路径
vocab_file_ner = os.path.join(dataneeded_dir, 'vocab.txt')
onnx_path_ner = os.path.join(dataneeded_dir, 'chineseNER.onnx')
label_list_file_ner = os.path.join(dataneeded_dir, 'label_list.txt')


def preprocess_message(line):
    """对消息内容进行预处理，将所有字符（包括标点符号和特殊符号）用空格隔开"""
    # 在所有非空白字符之间插入空格
    line = re.sub(r'(?<=\S)(?=\S)', ' ', line)
    # 去除多余的空格
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def get_label_list_ner(file_path):
    """获取标签列表"""
    with open(file_path, 'r', encoding='utf-8') as file:
        label_list = [line.strip() for line in file]
    return label_list


def most_frequent(arr):
    """找到数组中最频繁的元素"""
    counts = np.bincount(arr)
    return np.argmax(counts)


def load_chat_data(file_path):
    """加载聊天数据"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)
    return chat_data


def save_chat_data(file_path, chat_data):
    """保存聊天数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=4)


def vocab_to_dict_key_token(vocab_file):
    """将 vocab 文件转换为字典"""
    vocab_dict = {}
    try:
        with open(vocab_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                if len(line) > 0:
                    vocab_dict[line] = len(vocab_dict)
    except UnicodeDecodeError:
        print(f"无法使用 utf-8 编码解码文件: {vocab_file}")
        try:
            with open(vocab_file, 'r', encoding='gb18030') as reader:
                for line in reader:
                    line = line.strip()
                    if len(line) > 0:
                        vocab_dict[line] = len(vocab_dict)
        except Exception as e:
            print(f"读取文件时出错: {e}")
    return vocab_dict


def perform_entity_recognition(message):
    """对单条消息进行实体识别"""
    try:
        if not onnx_session_ner or not tokenizer_ner:
            return []
        
        # 对消息内容进行预处理
        preprocessed_message = preprocess_message(message)
        text_for_ner = preprocessed_message
        
        # 创建 InputExample 对象
        label = "O " * len(text_for_ner.split())
        example_ner = InputExample(text=text_for_ner, label=label)
        
        # 转换为特征
        feature_ner = convert_single_example(
            ex_index=0,
            example=example_ner,
            label_list=label_list_ner,
            vocab_file=vocab_file_ner,
            mode="",
            output_dir="",
            max_seq_length=max_seq_len_ner,
            tokenizer=tokenizer_ner
        )
        
        # 准备输入数据
        input_ids_ner = np.array([feature_ner.input_ids], dtype=np.int32)
        input_mask_ner = np.array([feature_ner.input_mask], dtype=np.int32)
        input_segment_ner = np.array([feature_ner.segment_ids], dtype=np.int32)
        label_ids_ner = np.array([feature_ner.label_ids], dtype=np.int32)
        real_seq_length_ner = len(text_for_ner.split()) + 2
        real_seq_length_tensor_ner = np.array([real_seq_length_ner], dtype=np.int32)
        
        # 进行预测
        logits_ner = onnx_session_ner.run(
            None,
            {
                "input_ids": input_ids_ner,
                "input_mask": input_mask_ner,
                "token_type_id": input_segment_ner,
                "label_ids": label_ids_ner,
                "real_seq_length": real_seq_length_tensor_ner,
            },
        )
        
        # 后处理逻辑
        logits_ner = logits_ner[0]
        best_pred_ner = []
        for i in range(real_seq_length_ner):
            pred_y = logits_ner[i]
            pred_y = most_frequent(pred_y)
            # 确保 pred_y 在 label_list_ner 的范围内
            if pred_y >= len(label_list_ner):
                pred_y = 0  # 默认设置为 0（O 标签）
            best_tag = label_list_ner[pred_y]
            best_pred_ner.append(best_tag)
        
        # 生成实体识别结果
        entities = []
        current_entity = None
        text_tokens_ner = text_for_ner.split()
        for idx, (token, pred) in enumerate(zip(text_tokens_ner, best_pred_ner[1:-1])):
            if pred.startswith("B_"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "entity": pred[2:],
                    "range": [idx, idx + 1]
                }
            elif pred.startswith("I_") and current_entity and current_entity["entity"] == pred[2:]:
                current_entity["range"][1] = idx + 1
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        if current_entity:
            entities.append(current_entity)
        
        # 计算字符级的范围
        char_entities = []
        current_char_index = 0
        for entity in entities:
            entity_type = entity["entity"]
            start_token = entity["range"][0]
            end_token = entity["range"][1]
            start_char = sum(len(text_tokens_ner[i]) for i in range(start_token))
            end_char = sum(len(text_tokens_ner[i]) for i in range(end_token))
            char_entities.append({
                "entity": entity_type,
                "range": [start_char, end_char]
            })
        
        return char_entities
    
    except Exception as e:
        print(f"实体识别出错: {e}")
        return []


@app.route('/perform_entity_recognition', methods=['POST'])
def perform_entity_recognition_route():
    """对 user_chat 文件夹中的所有消息文件进行实体识别"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        user_chat_path = os.path.join(current_dir, user_chat_dir)
        
        if not os.path.exists(user_chat_path):
            return jsonify({"result": "No chat files found"}), 200
        
        for file_name in os.listdir(user_chat_path):
            if file_name.endswith("_chat_results.json"):
                file_path = os.path.join(user_chat_path, file_name)
                chat_data = load_chat_data(file_path)
                
                for msg in chat_data:
                    if "message" in msg:
                        # 执行实体识别
                        entities = perform_entity_recognition(msg["message"])
                        msg["entities"] = entities
                
                # 保存更新后的聊天数据
                save_chat_data(file_path, chat_data)
        
        return jsonify({"result": "Entity recognition completed successfully"})
    
    except Exception as e:
        return jsonify({"error": f"Error performing entity recognition: {str(e)}"}), 500
    



# 对话分析部分
onnx_path = os.path.join(dataneeded_dir, 'quantiDCE.onnx')
tokenizer_pth = os.path.join(dataneeded_dir, 'models--google-bert--bert-base-chinese')

onnx_session = None
tokenizer = None
relatedscore = 0.6
maxcpmparetimes = 100
maxrelationcount = 5


def encode_ctx_res_pair(context, response: str, tokenizer):
    context = ' '.join(context)  
    tokenizer_outputs = tokenizer(
        text=context, text_pair=response,
        return_tensors='np', truncation=True,
        padding='max_length', max_length=128)
    
    input_ids = tokenizer_outputs['input_ids'].astype(np.int32)
    token_type_ids = tokenizer_outputs['token_type_ids'].astype(np.int32)
    attention_mask = tokenizer_outputs['attention_mask'].astype(np.int32)

    return input_ids, token_type_ids, attention_mask

def analyze_text_relationship(text1, text2):
    global onnx_session, tokenizer
    if not onnx_session or not tokenizer:
        return 0.0  

    input_ids, token_type_ids, attention_mask = encode_ctx_res_pair([text1], text2, tokenizer)

    # 进行预测
    inputs = {
        "input_ids": input_ids,
        "token_type_id": token_type_ids,  # 与ONNX模型期望的输入名称匹配
        "input_mask": attention_mask       # 与ONNX模型期望的输入名称匹配
    }
    logits = onnx_session.run(None, inputs)

    # 调试用代码，检查模型输出
    # print(f"logits type: {type(logits)}")
    # print(f"logits shape: {logits[0].shape}")
    # print(f"logits values: {logits[0].ravel()}")  # ravel() 将数组展平为 1 维
    
    # 提取单个值并转换为浮点数
    score = float(logits[0].item(0))  # 使用 item() 提取单个值
    return score

def load_chat_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)
    return chat_data

def save_chat_data(file_path, chat_data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=4)

def analyze_message_relationships(file_path):
    """分析对话文件的消息关联性，并更新相关消息"""
    try:
        chat_data = load_chat_data(file_path)
        if not chat_data:
            return False


        # 为每条消息初始化相关消息列表
        for msg in chat_data:
            if "related_before" not in msg:
                msg["related_before"] = []
            if "related_after" not in msg:
                msg["related_after"] = []

        # 创建一个字典来存储向后的关联关系
        related_after_dict = {}

        # 遍历每个消息，分析向前的相关性并记录向后的关联关系
        for i, msg in enumerate(chat_data):
            # 跳过 SYS 发送者、"[图片]" 或 "[动画表情]" 消息
            if msg.get("sender") == "SYS" or msg.get("message") in ["[图片]", "[动画表情]"]:
                continue

            if msg.get("sender") == "SYS" or msg.get("message") in ["[图片]", "[动画表情]"]:
                continue

            count = 0
            found = 0
            j = i - 1

            while j >= 0 and count < maxcpmparetimes and found < maxrelationcount:
                if chat_data[j].get("sender") == "SYS" or chat_data[j].get("message") in ["[图片]", "[动画表情]"]:
                    j -= 1
                    count += 1
                    continue

                score = analyze_text_relationship(chat_data[j]["message"], msg["message"])
                if score > relatedscore:
                    # 检查是否已存在该相关消息
                    id_exists_before = any(rel_msg["id"] == chat_data[j]["id"] for rel_msg in msg["related_before"])
                    if not id_exists_before:
                        msg["related_before"].append({
                            "id": chat_data[j]["id"],
                            "score": score
                        })

                    # 记录向后的关联关系
                    if chat_data[j]["id"] not in related_after_dict:
                        related_after_dict[chat_data[j]["id"]] = []
                    related_after_dict[chat_data[j]["id"]].append({
                        "id": msg["id"],
                        "score": score
                    })
                    found += 1

                count += 1
                j -= 1

        # 更新向后的相关消息
        for msg in chat_data:
            if msg["id"] in related_after_dict:
                # 过滤掉已存在的相关消息
                existing_related_after = []
                for rel_msg in related_after_dict[msg["id"]]:
                    if not any(r["id"] == rel_msg["id"] for r in msg["related_after"]):
                        existing_related_after.append(rel_msg)
                msg["related_after"] = existing_related_after
                msg["related_after"].sort(key=lambda x: x["score"], reverse=True)

        save_chat_data(file_path, chat_data)
        return True
    except Exception as e:
        print(f"分析消息关联性时出错: {str(e)}")
        return False

@app.route('/analyze_relationships', methods=['POST'])
def analyze_relationships():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        user_chat_path = os.path.join(current_dir, user_chat_dir)
        
        if not os.path.exists(user_chat_path):
            return jsonify({"result": "No chat files found"}), 200
        
        for file_name in os.listdir(user_chat_path):
            if file_name.endswith("_chat_results.json"):
                file_path = os.path.join(user_chat_path, file_name)
                analyze_message_relationships(file_path)
        
        return jsonify({"result": "Message relationships analyzed successfully"})
    
    except Exception as e:
        return jsonify({"error": f"Error analyzing message relationships: {str(e)}"}), 500


def calculate_topic_tightness(chat_data, msg_id, relatedscore):
    """递归计算话题的紧密度和消息数量"""
    msg = next((m for m in chat_data if m["id"] == msg_id), None)
    if not msg:
        return relatedscore, 1

    total_tightness = 0
    total_quantity = 0

    for rel_after in msg.get("related_after", []):
        child_tightness, child_quantity = calculate_topic_tightness(chat_data, rel_after["id"], relatedscore)
        total_tightness += rel_after["score"] * child_tightness
        total_quantity += child_quantity

    return total_tightness, total_quantity

def analyze_topics(file_path):
    """分析所有消息，确定话题开始并计算紧密度和消息数量"""
    try:
        chat_data = load_chat_data(file_path)
        if not chat_data:
            return False

        relatedscore = 0.5  # 相关性阈值

        # 初始化所有消息的 topic 相关字段
        for msg in chat_data:
            msg["topic_start"] = False
            msg["topic_quantity"] = 0
            msg["topic_Tightness"] = 0.0
            msg["topic_message_ids"] = []  # 新增字段，用于存储话题下所有消息ID

        # 遍历所有消息，确定话题开始并计算紧密度和数量
        for msg in chat_data:
            # 如果有 related_after 但没有 related_before，则标记为话题开始
            if msg.get("related_after") and not msg.get("related_before"):
                msg["topic_start"] = True

                # 计算紧密度和消息数量
                tightness, quantity = calculate_topic_tightness(chat_data, msg["id"], relatedscore)
                msg["topic_Tightness"] = tightness
                msg["topic_quantity"] = quantity

                # 收集话题下的所有消息ID
                collected_ids = set()  # 使用集合来避免重复
                collect_topic_messages(chat_data, msg["id"], collected_ids)
                msg["topic_message_ids"] = list(collected_ids)  # 转换为列表

        save_chat_data(file_path, chat_data)
        return True
    except Exception as e:
        print(f"分析话题时出错: {str(e)}")
        return False

def collect_topic_messages(chat_data, msg_id, collected_ids):
    """递归收集话题下的所有消息ID"""
    msg = next((m for m in chat_data if m["id"] == msg_id), None)
    if not msg:
        return

    collected_ids.add(msg["id"])  # 使用集合的 add 方法避免重复

    for rel_after in msg.get("related_after", []):
        collect_topic_messages(chat_data, rel_after["id"], collected_ids)

@app.route('/analyze_topics', methods=['POST'])
def analyze_topics_route():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        user_chat_path = os.path.join(current_dir, user_chat_dir)
        
        if not os.path.exists(user_chat_path):
            return jsonify({"result": "No chat files found"}), 200
        
        for file_name in os.listdir(user_chat_path):
            if file_name.endswith("_chat_results.json"):
                file_path = os.path.join(user_chat_path, file_name)
                analyze_topics(file_path)
        
        return jsonify({"result": "Topics analyzed successfully"})
    
    except Exception as e:
        return jsonify({"error": f"Error analyzing topics: {str(e)}"}), 500


def clear_analyses(file_path):
    """清除消息中的实体识别、相关性和话题分析信息"""
    try:
        chat_data = load_chat_data(file_path)
        if not chat_data:
            return False

        for msg in chat_data:
            # 清除实体识别信息
            if "entities" in msg:
                del msg["entities"]

            # 清除相关性分析信息
            if "related_before" in msg:
                del msg["related_before"]
            if "related_after" in msg:
                del msg["related_after"]

            # 清除话题分析信息
            if "topic_start" in msg:
                del msg["topic_start"]
            if "topic_quantity" in msg:
                del msg["topic_quantity"]
            if "topic_Tightness" in msg:
                del msg["topic_Tightness"]
            if "topic_message_ids" in msg:
                del msg["topic_message_ids"]

        save_chat_data(file_path, chat_data)
        return True
    except Exception as e:
        print(f"清除分析信息时出错: {str(e)}")
        return False

@app.route('/clear_analyses', methods=['POST'])
def clear_analyses_route():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        user_chat_path = os.path.join(current_dir, user_chat_dir)
        
        if not os.path.exists(user_chat_path):
            return jsonify({"result": "No chat files found"}), 200
        
        for file_name in os.listdir(user_chat_path):
            if file_name.endswith("_chat_results.json"):
                file_path = os.path.join(user_chat_path, file_name)
                clear_analyses(file_path)
        
        return jsonify({"result": "Analyses cleared successfully"})
    
    except Exception as e:
        return jsonify({"error": f"Error clearing analyses: {str(e)}"}), 500
    

def update_topic_visibility(file_path, target_id, visible):
    """更新指定话题及其关联消息的可见性"""
    try:
        chat_data = load_chat_data(file_path)
        if not chat_data:
            return False

        target_msg = None
        for msg in chat_data:
            if msg.get("id") == target_id:
                target_msg = msg
                break

        if not target_msg:
            return False

        if not target_msg.get("topic_start"):
            return False

        # 收集话题下的所有消息ID
        collected_ids = set()
        collected_ids.add(target_id)
        for rel_after in target_msg.get("related_after", []):
            def collect_messages(msg_id):
                msg = next((m for m in chat_data if m["id"] == msg_id), None)
                if msg:
                    collected_ids.add(msg_id)
                    for rel in msg.get("related_after", []):
                        collect_messages(rel["id"])
            collect_messages(target_id)

        # 更新可见性
        for msg in chat_data:
            if msg["id"] in collected_ids:
                msg["visible"] = visible
            # else:
            #     msg["visible"] = False

        save_chat_data(file_path, chat_data)
        return True
    except Exception as e:
        print(f"更新话题可见性时出错: {str(e)}")
        return False


def update_all_messages_visible(file_path, visible):
    """更新指定对话窗口中所有消息的可见性"""
    try:
        chat_data = load_chat_data(file_path)
        if not chat_data:
            return False

        for msg in chat_data:
            msg["visible"] = visible

        save_chat_data(file_path, chat_data)
        return True
    except Exception as e:
        print(f"更新所有消息可见性时出错: {str(e)}")
        return False


@app.route('/update_topic_visibility', methods=['POST'])
def update_topic_visibility_route():
    """更新指定话题及其关联消息的可见性"""
    try:
        data = request.json
        if not data or 'name' not in data or 'id' not in data:
            return jsonify({"error": "Invalid input format"}), 400

        name = data['name']
        msg_id = data['id']
        visible = data.get('visible', True)

        user_chat_path = os.path.join(current_dir, user_chat_dir.lstrip("/"))
        if not os.path.exists(user_chat_path):
            return jsonify({"result": f"No chat found for {name}"}), 404

        file_path = os.path.join(user_chat_path, f"{name}_chat_results.json")
        if not os.path.exists(file_path):
            return jsonify({"result": f"No chat found for {name}"}), 404

        success = update_topic_visibility(file_path, msg_id, visible)
        if not success:
            return jsonify({"result": f"Failed to update visibility for topic starting at id {msg_id}"}), 500

        updated_chat_data = load_chat_data(file_path)
        return jsonify(updated_chat_data), 200

    except Exception as e:
        return jsonify({"error": f"Error updating topic visibility: {str(e)}"}), 500


@app.route('/update_all_visible', methods=['POST'])
def update_all_visible_route():
    """更新指定对话窗口中所有消息的可见性"""
    try:
        data = request.json
        if not data or 'name' not in data:
            return jsonify({"error": "Invalid input format"}), 400

        name = data['name']
        visible = data.get('visible', True)

        user_chat_path = os.path.join(current_dir, user_chat_dir.lstrip("/"))
        if not os.path.exists(user_chat_path):
            return jsonify({"result": f"No chat found for {name}"}), 404

        file_path = os.path.join(user_chat_path, f"{name}_chat_results.json")
        if not os.path.exists(file_path):
            return jsonify({"result": f"No chat found for {name}"}), 404

        success = update_all_messages_visible(file_path, visible)
        if not success:
            return jsonify({"result": f"Failed to update visibility for all messages in chat {name}"}), 500

        updated_chat_data = load_chat_data(file_path)
        return jsonify(updated_chat_data), 200

    except Exception as e:
        return jsonify({"error": f"Error updating all messages visibility: {str(e)}"}), 500
    

if __name__ == '__main__':
    # 初始化实体识别相关组件
    label_list_ner = get_label_list_ner(label_list_file_ner)
    tokenizer_ner = tokenization.FullTokenizer(vocab_file=vocab_file_ner, do_lower_case=True)
    tokenizer_ner.vocab_dict = vocab_to_dict_key_token(vocab_file_ner)
    tag_to_index_ner = convert_labels_to_index(label_list_ner)
    onnx_model_ner = onnx.load(onnx_path_ner)
    onnx.checker.check_model(onnx_model_ner)
    onnx_session_ner = onnxruntime.InferenceSession(onnx_path_ner)

    # 初始化对话分析相关组件
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, local_files_only=True)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_session = onnxruntime.InferenceSession(onnx_path)

    app.run(host='0.0.0.0', port=5000)