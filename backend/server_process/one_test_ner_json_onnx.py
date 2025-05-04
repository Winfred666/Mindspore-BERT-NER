from flask import Flask, request, jsonify
import os
import json
import re
import numpy as np
# from src.generate_mindrecord.generate_chinesener_mindrecord import convert_single_example, InputExample
# import src.tokenization as tokenization

from src.dependency import convert_single_example, InputExample
import src.tokenization as tokenization

import onnx
import onnxruntime
import collections

app = Flask(__name__)


# use BMES representation, however can use other to do it.
def convert_labels_to_index(label_list):
    """
    Convert label_list to indices for NER task.
    """
    label2id = collections.OrderedDict()
    label2id["O"] = 0
    # prefix = ["S_", "B_", "M_", "E_"]
    index = 0
    for label in label_list:
        if(label == "O"):
            continue
        # for pre in prefix:
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
vocab_file_ner = "/data/songjh/bert/vocab.txt"
onnx_path_ner = "/data/songjh/ONNX/chineseNER.onnx"  # 指定ONNX模型路径
label_list_file_ner = "/data/songjh/mindrecord/literature_NER/label_list.txt"

def preprocess_message(line):
    """对消息内容进行预处理，将所有字符（包括标点符号和特殊符号）用空格隔开"""
    # 在所有非空白字符之间插入空格
    line = re.sub(r'(?<=\S)(?=\S)', ' ', line)
    # 去除多余的空格
    line = re.sub(r'\s+', ' ', line).strip()
    return line

def get_label_list_ner(file_path):
    """获取标签列表"""
    with open(file_path, 'r') as file:
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

if __name__ == '__main__':
    # 初始化实体识别相关组件
    label_list_ner = get_label_list_ner(label_list_file_ner)
    tokenizer_ner = tokenization.FullTokenizer(vocab_file=vocab_file_ner, do_lower_case=True)
    tag_to_index_ner = convert_labels_to_index(label_list_ner)
    
    # 加载ONNX模型
    onnx_model = onnx.load(onnx_path_ner)
    onnx.checker.check_model(onnx_model)
    onnx_session_ner = onnxruntime.InferenceSession(onnx_path_ner)

    app.run(host='0.0.0.0', port=5000)