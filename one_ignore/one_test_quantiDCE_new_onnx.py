# from src.finetune_eval_model import BertQuantiDCEModel
# from src.model_utils.config import bert_net_cfg

import pickle
import os

import numpy as np
import json

from flask import Flask, request, jsonify

from transformers import AutoTokenizer

import onnx
import onnxruntime

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 相对路径到 dataneeded 目录
dataneeded_dir = os.path.join(current_dir, 'dataneeded')

app = Flask(__name__)
user_chat_dir = "user_chat"

onnx_session = None
tokenizer = None
relatedscore = 0.5
maxcpmparetimes = 100
maxrelationcount = 5

onnx_path = os.path.join(dataneeded_dir, 'quantiDCE.onnx')  # 指定ONNX模型路径
tokenizer_pth = os.path.join(dataneeded_dir, 'models--google-bert--bert-base-chinese')

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
    try:
        chat_data = load_chat_data(file_path)
        if not chat_data:
            return False

        for i, msg in enumerate(chat_data):
            # 跳过 SYS 发送者、"[图片]" 或 "[动画表情]" 消息
            if msg.get("sender") == "SYS" or msg.get("message") in ["[图片]", "[动画表情]"]:
                continue

            if "related_messages" in msg and msg["related_messages"]:
                continue

            if "related_messages" not in msg:
                msg["related_messages"] = []

            count = 0
            found = 0
            j = i - 1

            while j >= 0 and count < maxcpmparetimes and found < maxrelationcount:
                if "message" not in chat_data[j]:
                    j -= 1
                    count += 1
                    continue

                # 跳过 SYS 发送者、"[图片]" 或 "[动画表情]" 消息
                if chat_data[j].get("sender") == "SYS" or chat_data[j].get("message") in ["[图片]", "[动画表情]"]:
                    j -= 1
                    count += 1
                    continue

                score = analyze_text_relationship(chat_data[j]["message"], msg["message"])
                if score > relatedscore:  
                    msg["related_messages"].append({
                        "id": chat_data[j]["id"],
                        "score": score
                    })
                    found += 1

                count += 1
                j -= 1

            msg["related_messages"].sort(key=lambda x: x["score"], reverse=True)

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

if __name__ == "__main__":
    # 初始化相关组件
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, local_files_only=True)
    
    # 加载ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_session = onnxruntime.InferenceSession(onnx_path)

    app.run(host='0.0.0.0', port=5000)