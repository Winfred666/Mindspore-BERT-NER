import onnxruntime
import numpy as np
from transformers import AutoTokenizer

# 输入示例
context_example = ["你好！", "请问图书馆怎么走？"]
response_example = "我觉得学校饭堂的菜不错"

# 模型和tokenizer路径
tokenizer_path = "/data/songjh/pretrained/models--google-bert--bert-base-chinese"
onnx_model_path = "/data/songjh/ONNX/quantiDCE.onnx"

# 编码函数，将文本对编码为模型输入格式
def encode_ctx_res_pair(context, response, tokenizer):
    """
    Encodes the given context-response pair into ids.
    """
    context = ' '.join(context)  # 将列表转换为字符串
    tokenizer_outputs = tokenizer(
        text=context,
        text_pair=response,
        return_tensors='np',  # 返回NumPy格式
        truncation=True,
        padding='max_length',
        max_length=128
    )
    return {
        'input_ids': tokenizer_outputs['input_ids'],
        'token_type_ids': tokenizer_outputs['token_type_ids'],
        'attention_mask': tokenizer_outputs['attention_mask']
    }

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

# 对输入文本进行编码
inputs = encode_ctx_res_pair(context_example, response_example, tokenizer)

# 加载ONNX模型
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# 获取模型输入名称
input_names = {input_name: inputs[input_name] for input_name in ort_session.get_inputs()}

# 执行推理
ort_outs = ort_session.run(None, input_names)

# 输出推理结果
print("ONNX Runtime inference results:", ort_outs)