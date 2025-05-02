import os
import numpy as np
from src.generate_mindrecord.generate_chinesener_mindrecord import convert_single_example, InputExample
import src.tokenization as tokenization
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.utils import convert_labels_to_index
from src.model_utils.config import bert_net_cfg
from src.bert_for_finetune import BertNER
from src.bert_model import BertConfig

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

from mindspore import context

import re
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# 全局变量
label_list_ner = []
net_for_test_ner = None
model_ner = None
tokenizer_ner = None
tag_to_index_ner = {}
max_seq_len_ner = 128
vocab_file_ner = "/data/songjh/bert/vocab.txt"
check_point_ner = "/data/songjh/bert/finetuned/chi_literature/1/ner-2_6041.ckpt"
label_list_file_ner = "/data/songjh/mindrecord/literature_NER/label_list.txt"

def get_label_list_ner(file_path):
    with open(file_path, 'r') as file:
        label_list = [line.strip() for line in file]
    return label_list

def most_frequent(arr):
    counts = np.bincount(arr)
    return np.argmax(counts)

def encode_text(text, tokenizer, max_seq_len):
    text = text.split()
    tokens = []
    for word in text:
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    tokens = ["[CLS]"] + tokens[:max_seq_len-2] + ["[SEP]"]
    input_ids = tokenizer.tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    return input_ids, input_mask, segment_ids

def preprocess_message(line):
    # 将连续数字拆分为单个数字并在它们之间插入空格
    line = re.sub(r'(\d)', r'\1 ', line)
    # 将每个单词拆分为单个字母并在字母之间插入空格
    line = re.sub(r'(\w)(?=\w)', r'\1 ', line)
    # 去除多余的空格
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def perform_ner_on_text_file(file_path):
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"文件 {file_path} 不存在"

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 处理文件内容
        results = []
        current_sender = None
        current_message = []
        line_count = 0  # 用于跟踪行号

        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_ner, do_lower_case=True)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_count += 1

            # 奇数行是发送者，偶数行是消息内容
            if line_count % 2 == 1:  # 奇数行
                current_sender = line
                current_message = []
            else:  # 偶数行
                # 如果发送者是 SYS，则跳过 NER 处理
                if current_sender == "SYS :":
                    continue

                # 对消息内容进行预处理
                preprocessed_line = preprocess_message(line)
                # 对消息内容进行分词，并在单词之间插入空格
                tokenized_message = tokenizer.tokenize(preprocessed_line)
                text_for_ner = ' '.join(tokenized_message)
                current_message = tokenized_message

                try:
                    # 调用 NER 识别逻辑
                    example_ner = InputExample(text=text_for_ner, label="O " * len(text_for_ner.split()))
                    
                    # 调用 convert_single_example 函数进行编码
                    feature_ner = convert_single_example(
                        ex_index=0,
                        example=example_ner,
                        label_list=label_list_ner,
                        vocab_file=vocab_file_ner,
                        mode="",
                        output_dir="",
                        max_seq_length=max_seq_len_ner,
                        tokenizer=tokenizer
                    )
                    
                    # 准备输入数据
                    input_ids_ner = Tensor([feature_ner.input_ids], dtype=mstype.int32)
                    input_mask_ner = Tensor([feature_ner.input_mask], dtype=mstype.int32)
                    input_segment_ner = Tensor([feature_ner.segment_ids], dtype=mstype.int32)
                    label_ids_ner = Tensor([feature_ner.label_ids], dtype=mstype.int32)
                    real_seq_length_ner = len(text_for_ner.split()) + 2
                    real_seq_length_tensor_ner = Tensor([real_seq_length_ner], dtype=mstype.int32)
                    
                    # 进行预测
                    logits_ner = model_ner.predict(input_ids_ner, input_mask_ner, input_segment_ner, label_ids_ner, real_seq_length_tensor_ner)
                    
                    # 后处理逻辑
                    logits_ner = logits_ner[0]
                    best_pred_ner = []
                    for i in range(real_seq_length_ner):
                        pred_y = logits_ner[i][0][0]
                        pred_y = most_frequent(pred_y)
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
                    for entity in entities:
                        entity_type = entity["entity"]
                        start_token = entity["range"][0]
                        end_token = entity["range"][1]
                        start_char = sum(len(text_tokens_ner[i])  for i in range(start_token))  # +1 考虑空格
                        end_char = sum(len(text_tokens_ner[i])  for i in range(end_token))  # +1 考虑空格
                        char_entities.append({
                            "entity": entity_type,
                            "range": [start_char, end_char]
                        })
                    
                    # 保存结果
                    results.append({
                        "sender": current_sender,
                        "message": text_for_ner,
                        "entities": char_entities
                    })
                
                except Exception as ner_e:
                    print(f"NER处理出错: {ner_e}")

        # 将结果保存到文件
        output_file_path = os.path.join(os.path.dirname(file_path), "ner_results.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        return f"NER识别结果已保存到 {output_file_path}"

    except Exception as e:
        return f"处理文件时出错: {e}"

@app.route('/perform_ner', methods=['POST'])
def perform_ner():
    data = request.json
    if not data or 'file_path' not in data:
        return jsonify({"error": "Invalid input format"}), 400
    file_path = data['file_path']
    result = perform_ner_on_text_file(file_path)
    return jsonify({"result": result})

if __name__ == '__main__':
    # 初始化 NER 相关组件
    label_list_ner = get_label_list_ner(label_list_file_ner)
    tokenizer_ner = tokenization.FullTokenizer(vocab_file=vocab_file_ner, do_lower_case=True)
    tag_to_index_ner = convert_labels_to_index(label_list_ner)
    max_val = max(tag_to_index_ner.values())
    tag_to_index_ner["<START>"] = max_val + 1
    tag_to_index_ner["<STOP>"] = max_val + 2
    number_labels_ner = len(tag_to_index_ner)
    label_list_ner.extend(["<START>", "<STOP>"])
    net_for_test_ner = BertNER(bert_net_cfg, batch_size=1, is_training=False, num_labels=number_labels_ner, with_lstm=True, use_crf=True, tag_to_index=tag_to_index_ner)
    net_for_test_ner.set_train(False)
    param_dict_ner = load_checkpoint(check_point_ner)
    load_param_into_net(net_for_test_ner, param_dict_ner)
    model_ner = Model(net_for_test_ner)
    app.run(host='0.0.0.0', port=5000)