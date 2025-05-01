# get quantiDEC model only for evaluation:

context_example = ["你好！", "请问图书馆怎么走？"]
response_example = "我觉得学校饭堂的菜不错"

# When there are no network to huggingface, use local file.
tokenizer_path = "/data/songjh/pretrained/models--google-bert--bert-base-chinese"
dict_ckpt_path = "/data/songjh/bert/finetuned/quantiDCE/chinese_finetuned.pkl"

export_ONNX_name = "/data/songjh/ONNX/quantiDCE.onnx"

need_export_ONNX = False

# For english, vocabulary size is bigger, need to change config.vocab_size
isEnglish = False

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

app = Flask(__name__)

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

network = None
model = None
tokenizer = None

def get_mindspore_ckpt_path():
    global dict_ckpt_path
    mindspore_ckpt = dict_ckpt_path[:-3] + "ckpt"
    
    if os.path.isfile(mindspore_ckpt):
        return mindspore_ckpt
    
    # get the param that the model needed
    loaded_dict = None 
    with open(dict_ckpt_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    
    # replace * in the dict to generate the mapping from 0 to 11 layers.
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

    # replace all params:
    new_params_list = []
    for old_key in param_map.keys():
        new_key = param_map[old_key]
        old_val = loaded_dict.pop(old_key)
        new_params_list.append({"name": new_key, "data": Tensor(old_val)})
    ms.save_checkpoint(new_params_list, mindspore_ckpt)
    return mindspore_ckpt


def encode_ctx_res_pair(context, response: str, tokenizer):
    """
    Encodes the given context-response pair into ids.
    """
    context = ' '.join(context)  # transfer list to one str.

    tokenizer_outputs = tokenizer(
        text=context, text_pair=response,
        return_tensors='np', truncation=True,
        padding='max_length', max_length=128)
    
    # should not batch it, because not training now.
    input_ids = Tensor(tokenizer_outputs['input_ids'], dtype=mstype.int32)
    token_type_ids = Tensor(tokenizer_outputs['token_type_ids'], dtype=mstype.int32)
    attention_mask = Tensor(tokenizer_outputs['attention_mask'], dtype=mstype.int32)

    return input_ids, token_type_ids, attention_mask

@app.route('/predict', methods=['POST'])
def predict():
    global network, model, tokenizer
    if not network or not model or not tokenizer:
        return jsonify({"error": "Model not initialized"}), 500
    
    data = request.json
    if not data or 'context' not in data or 'response' not in data:
        return jsonify({"error": "Invalid input format"}), 400
    
    context = data['context']
    response = data['response']

    input_ids, token_type_ids, attention_mask = encode_ctx_res_pair(context, response, tokenizer)

    # the score range is sigmoid output, that is, (0,1)
    score = model.predict(input_ids, token_type_ids, attention_mask)
    score = score[0][0]
    
    # for 1 - 5 rank score.
    # score = round(score * 4 + 1, 2)
    result = {
        'context': context,
        'response': response,
        'QuantiDCE score': float(score)
    }
    return jsonify(result)

@app.route('/export', methods=['GET'])
def export_model():
    global network, model
    if not network or not model:
        return jsonify({"error": "Model not initialized"}), 500
    
    input_ids, token_type_ids, attention_mask = encode_ctx_res_pair(context_example, response_example, tokenizer)
    input_format_ONNX = [input_ids, token_type_ids, attention_mask]
    try:
        export(network, *input_format_ONNX, file_name=export_ONNX_name, file_format="ONNX")
        return jsonify({"message": f"Model exported to {export_ONNX_name} successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    bert_net_cfg.vocab_size = 21128
    network = BertQuantiDCEModel(bert_net_cfg, is_training=False)
    network.set_train(False)
    
    ckpt_path = get_mindspore_ckpt_path()
    ms.load_checkpoint(ckpt_path, network)
    model = Model(network)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, clean_up_tokenization_spaces=True)
    
    app.run(host='0.0.0.0', port=5000)