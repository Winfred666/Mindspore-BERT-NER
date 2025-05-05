# get quantiDEC model only for evalidation:

context_example = ["好久不见！","年轻人，风还在吹吗？"]
response_example = "是的，永不停息。"


# When there are no network to huggingface, use local file.
tokenizer_path = "/data/songjh/pretrained/models--google-bert--bert-base-chinese"
dict_ckpt_path = "/data/songjh/bert/finetuned/quantiDCE/chinese_finetuned.pkl"

export_ONNX_name = "/data/songjh/ONNX/quantiDCE.onnx"

need_export_ONNX = False

# For english , vocabulary size is bigger, need to change config.vocab_size
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



def exportONNX(net, input_data):
    global export_ONNX_name
    # however this require the model to output a single Tensor instead of a tuple.
    export(net, *input_data, file_name=export_ONNX_name, file_format="ONNX")

param_map={
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
    "mlp.0.weight" : "mlp.0.weight",
    "mlp.2.weight" : "mlp.2.weight",
    "mlp.4.weight" : "mlp.4.weight",
    "mlp.0.bias" : "mlp.0.bias",
    "mlp.2.bias" : "mlp.2.bias",
    "mlp.4.bias" : "mlp.4.bias",

}


def get_mindspore_ckpt_path():
    global dict_ckpt_path
    mindspore_ckpt = dict_ckpt_path[:-3] + "ckpt"
    
    if(os.path.isfile(mindspore_ckpt)):
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
        for i in range(0,12):
            new_key = key.replace("*", str(i))
            new_value = param_map[key].replace("*",str(i))
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
        new_params_list.append({"name":new_key, "data": Tensor(old_val)})
    ms.save_checkpoint(new_params_list,mindspore_ckpt)
    return mindspore_ckpt


def encode_ctx_res_pair(context, response: str, tokenizer):
    """
    Encodes the given context-response pair into ids.
    """
    # context = ' '.join(context)
    # having two sentense(text_a and text_b), instead of using pytorch, 
    # here can also using mindspore tokenizer for sentense similarity.
    context = ' '.join(context) # transfer list to one str.

    tokenizer_outputs = tokenizer(
        text=context, text_pair=response,
        return_tensors='np', truncation=True,
        padding='max_length', max_length=128)
    
    # should not batch it, because not training now.
    input_ids = Tensor(tokenizer_outputs['input_ids'], dtype=mstype.int32)
    token_type_ids = Tensor(tokenizer_outputs['token_type_ids'], dtype=mstype.int32)
    attention_mask = Tensor(tokenizer_outputs['attention_mask'], dtype=mstype.int32)

    return input_ids, token_type_ids, attention_mask

# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params

if __name__ == "__main__":
    
    bert_net_cfg.vocab_size = 21128
    network = BertQuantiDCEModel(bert_net_cfg,is_training=False)
    network.set_train(False)
    
    ckpt_path = get_mindspore_ckpt_path()
    ms.load_checkpoint(ckpt_path,network)
    # when ckpt param fit in, we can now load ckpt
    model = Model(network)
    # only do one context-response pair to get one score.
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, clean_up_tokenization_spaces=True)
    input_ids, token_type_ids, attention_mask = encode_ctx_res_pair(context_example,response_example,tokenizer)

    # the score range is sigmoid output, that is, (0,1)
    score = model.predict(input_ids, token_type_ids, attention_mask)
    score = score[0][0]
    
    if(need_export_ONNX):
        input_format_ONNX = [input_ids, token_type_ids, attention_mask]
        exportONNX(network, input_format_ONNX)

    # for 1 - 5 rank score.
    # score = round(score * 4 + 1, 2)
    print('context:', context_example)
    print('response:', response_example)
    print('QuantiDCE score:', score)