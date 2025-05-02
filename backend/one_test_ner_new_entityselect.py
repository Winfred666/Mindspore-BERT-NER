import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import mindspore as ms
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

import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# 全局变量
label_list = []
net_for_test = None
model = None
tokenizer = None
tag_to_index = {}
max_seq_len = 128
vocab_file = "/data/songjh/bert/vocab.txt"
check_point = "/data/songjh/bert/finetuned/chi_literature/1/ner-2_6041.ckpt"
label_list_file = "/data/songjh/mindrecord/literature_NER/label_list.txt"

def get_label_list(file_path):
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

@app.route('/predict', methods=['POST'])
def predict():
    global net_for_test, model, tokenizer, label_list
    
    if not net_for_test or not model or not tokenizer:
        return jsonify({"error": "Model not initialized"}), 500

    data = request.json
    if not data or 'text' not in data or 'label' not in data:
        return jsonify({"error": "Invalid input format"}), 400

    text = data['text']
    label = data['label']

    # 统一标签格式
    label = label.replace("B-ORG", "B_Organization").replace("I-ORG", "I_Organization")
    label = label.replace("B-LOC", "B_Location").replace("I-LOC", "I_Location")
    label = label.replace("B-PER", "B_Person").replace("I-PER", "I_Person")

    # 创建 InputExample 对象
    example = InputExample(text=text, label=label)

    # 调用 convert_single_example 函数进行编码
    feature = convert_single_example(
        ex_index=0,
        example=example,
        label_list=label_list,
        vocab_file=vocab_file,
        mode="",
        output_dir="",
        max_seq_length=max_seq_len,
        tokenizer=tokenizer
    )

    # 准备输入数据
    input_ids = Tensor([feature.input_ids], dtype=mstype.int32)
    input_mask = Tensor([feature.input_mask], dtype=mstype.int32)
    input_segment = Tensor([feature.segment_ids], dtype=mstype.int32)
    label_ids = Tensor([feature.label_ids], dtype=mstype.int32)
    
    real_seq_length = len(text.split()) + 2
    real_seq_length_tensor = Tensor([real_seq_length], dtype=mstype.int32)

    # 进行预测
    logits = model.predict(input_ids, input_mask, input_segment, label_ids, real_seq_length_tensor)

    # 后处理逻辑保持不变
    logits = logits[0]
    best_pred = []
    for i in range(real_seq_length):
        pred_y = logits[i][0][0]
        pred_y = most_frequent(pred_y)
        best_tag = label_list[pred_y]
        best_pred.append(best_tag)

    # 生成实体识别结果
    entities = []
    current_entity = None
    for idx, (token, pred) in enumerate(zip(example.text.split(), best_pred[1:-1])):  # 跳过 [CLS] 和 [SEP]
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
    text_tokens = example.text.split()
    char_entities = []
    current_char_index = 0
    for entity in entities:
        entity_type = entity["entity"]
        start_token = entity["range"][0]
        end_token = entity["range"][1]
        start_char = sum(len(text_tokens[i])  for i in range(start_token))  # +1 是因为考虑到空格
        end_char = sum(len(text_tokens[i])  for i in range(end_token)) 
        char_entities.append({
            "entity": entity_type,
            "range": [start_char, end_char]
        })

    result = {
        'text': text,
        'label': label,
        'prediction': best_pred,
        'entities': char_entities
    }
    return jsonify(result)

@app.route('/export', methods=['GET'])
def export_model():
    if not net_for_test or not model:
        return jsonify({"error": "Model not initialized"}), 500
    return jsonify({"message": "Model exported successfully"})

if __name__ == "__main__":
    label_list = get_label_list(label_list_file)
    
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    
    tag_to_index = convert_labels_to_index(label_list)
    max_val = max(tag_to_index.values())
    tag_to_index["<START>"] = max_val + 1
    tag_to_index["<STOP>"] = max_val + 2
    number_labels = len(tag_to_index)
    label_list.extend(["<START>", "<STOP>"])

    net_for_test = BertNER(bert_net_cfg, batch_size=1, is_training=False, num_labels=number_labels, 
                          with_lstm=True, use_crf=True, tag_to_index=tag_to_index)
    net_for_test.set_train(False)
    param_dict = load_checkpoint(check_point)
    load_param_into_net(net_for_test, param_dict)
    model = Model(net_for_test)

    app.run(host='0.0.0.0', port=5000)