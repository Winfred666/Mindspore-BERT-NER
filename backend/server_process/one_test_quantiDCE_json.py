from src.finetune_eval_model import BertQuantiDCEModel
from src.model_utils.config import bert_net_cfg

import pickle
import os

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
import mindspore as ms

from mindspore import export
from transformers import AutoTokenizer

from flask import Flask, request, jsonify

import numpy as np
import json



app = Flask(__name__)
user_chat_dir = "user_chat"

network = None
model = None
tokenizer = None
relatedscore = 0.7
maxcpmparetimes = 100
maxrelationcount = 5

def get_mindspore_ckpt_path():
    global dict_ckpt_path
    mindspore_ckpt = dict_ckpt_path[:-3] + "ckpt"
    
    if os.path.exists(mindspore_ckpt):
        return mindspore_ckpt
    
    loaded_dict = None 
    with open(dict_ckpt_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    
    added_map = {}
    delete_key = []
    for key in param_map.keys():
        if "*" not in key:
            continue
        for i in range(0, 12):
            new_key = key.replace("*", str(i))
            new_value = param_map[key].replace("*", str(i))
            added_map[new_key] = new_value
        delete_key.append(key)
    
    for key in delete_key:
        param_map.pop(key)
    param_map.update(added_map)

    new_params_list = []
    for old_key in param_map.keys():
        new_key = param_map[old_key]
        old_val = loaded_dict.pop(old_key)
        new_params_list.append({"name": new_key, "data": Tensor(old_val)})
    ms.save_checkpoint(new_params_list, mindspore_ckpt)
    return mindspore_ckpt


def encode_ctx_res_pair(context, response: str, tokenizer):
    context = ' '.join(context)  
    tokenizer_outputs = tokenizer(
        text=context, text_pair=response,
        return_tensors='np', truncation=True,
        padding='max_length', max_length=128)
    
    input_ids = Tensor(tokenizer_outputs['input_ids'], dtype=mstype.int32)
    token_type_ids = Tensor(tokenizer_outputs['token_type_ids'], dtype=mstype.int32)
    attention_mask = Tensor(tokenizer_outputs['attention_mask'], dtype=mstype.int32)

    return input_ids, token_type_ids, attention_mask

def analyze_text_relationship(text1, text2):
    global network, model, tokenizer
    if not network or not model or not tokenizer:
        return 0.0  

    input_ids, token_type_ids, attention_mask = encode_ctx_res_pair([text1], text2, tokenizer)

    score = model.predict(input_ids, token_type_ids, attention_mask)
    score = score.asnumpy()  
    score = float(score[0][0])  

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

                score = analyze_text_relationship(msg["message"], chat_data[j]["message"])
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

param_map = {
    "backbone.embeddings.word_embeddings.weight": "bert.bert_embedding_lookup.embedding_table",
    "backbone.embeddings.position_embeddings.weight": "bert.bert_embedding_postprocessor.full_position_embedding.embedding_table",
    "backbone.embeddings.token_type_embeddings.weight": "bert.bert_embedding_postprocessor.token_type_embedding.embedding_table",
    "backbone.embeddings.LayerNorm.weight": "bert.bert_embedding_postprocessor.layernorm.gamma",
    "backbone.embeddings.LayerNorm.bias": "bert.bert_embedding_postprocessor.layernorm.beta",
    "backbone.encoder.layer.*.attention.self.query.weight": "bert.bert_encoder.layers.*.attention.attention.query_layer.weight",
    "backbone.encoder.layer.*.attention.self.query.bias": "bert.bert_encoder.layers.*.attention.attention.query_layer.bias",
    "backbone.encoder.layer.*.attention.self.key.weight": "bert.bert_encoder.layers.*.attention.attention.key_layer.weight",
    "backbone.encoder.layer.*.attention.self.key.bias": "bert.bert_encoder.layers.*.attention.attention.key_layer.bias",
    "backbone.encoder.layer.*.attention.self.value.weight": "bert.bert_encoder.layers.*.attention.attention.value_layer.weight",
    "backbone.encoder.layer.*.attention.self.value.bias": "bert.bert_encoder.layers.*.attention.attention.value_layer.bias",
    "backbone.encoder.layer.*.attention.output.dense.weight": "bert.bert_encoder.layers.*.attention.output.dense.weight",
    "backbone.encoder.layer.*.attention.output.dense.bias": "bert.bert_encoder.layers.*.attention.output.dense.bias",
    "backbone.encoder.layer.*.attention.output.LayerNorm.weight": "bert.bert_encoder.layers.*.attention.output.layernorm.gamma",
    "backbone.encoder.layer.*.attention.output.LayerNorm.bias": "bert.bert_encoder.layers.*.attention.output.layernorm.beta",
    "backbone.encoder.layer.*.intermediate.dense.weight": "bert.bert_encoder.layers.*.intermediate.weight",
    "backbone.encoder.layer.*.intermediate.dense.bias": "bert.bert_encoder.layers.*.intermediate.bias",
    "backbone.encoder.layer.*.output.dense.weight": "bert.bert_encoder.layers.*.output.dense.weight",
    "backbone.encoder.layer.*.output.dense.bias": "bert.bert_encoder.layers.*.output.dense.bias",
    "backbone.encoder.layer.*.output.LayerNorm.weight": "bert.bert_encoder.layers.*.output.layernorm.gamma",
    "backbone.encoder.layer.*.output.LayerNorm.bias": "bert.bert_encoder.layers.*.output.layernorm.beta",
    "backbone.pooler.dense.weight": "bert.dense.weight",
    "backbone.pooler.dense.bias": "bert.dense.bias",
    "mlp.0.weight": "mlp.0.weight",
    "mlp.2.weight": "mlp.2.weight",
    "mlp.4.weight": "mlp.4.weight",
    "mlp.0.bias": "mlp.0.bias",
    "mlp.2.bias": "mlp.2.bias",
    "mlp.4.bias": "mlp.4.bias",
}

dict_ckpt_path = "/data/songjh/bert/finetuned/quantiDCE/chinese_finetuned.pkl"
tokenizer_path = "/data/songjh/pretrained/models--google-bert--bert-base-chinese"

if __name__ == "__main__":
    from src.finetune_eval_model import BertQuantiDCEModel
    from src.model_utils.config import bert_net_cfg
    import pickle
    import mindspore as ms

    bert_net_cfg.vocab_size = 21128
    network = BertQuantiDCEModel(bert_net_cfg, is_training=False)
    network.set_train(False)
    
    ckpt_path = get_mindspore_ckpt_path()
    ms.load_checkpoint(ckpt_path, network)
    model = Model(network)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, clean_up_tokenization_spaces=True)
    
    app.run(host='0.0.0.0', port=5000)