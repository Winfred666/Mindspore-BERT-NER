import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import mindspore as ms
from src.generate_mindrecord.generate_chinesener_mindrecord import convert_single_example,InputExample
import src.tokenization as tokenization
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.utils import convert_labels_to_index
from src.model_utils.config import bert_net_cfg
# from src.finetune_eval_model import BertNERModel
from src.bert_for_finetune import BertNER
from src.bert_model import BertConfig

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

from mindspore import context

import numpy as np

import os

# If no argument is provided, set CUDA_VISIBLE_DEVICES to 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Optional: you can check the value
print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

example_1 = InputExample(
    text ="中 秋 假 期 ， 来 自 清 华 大 学 的 李 四 同 学 在 上 海 旅 游 ， 花 费 3 0 0 元 ， 买 了 一 袋 苹 果 和 一 本 图 书 。",
    label="B_Time I_Time I_Time I_Time O O O B-ORG I-ORG I-ORG I-ORG O B-PER I-PER O O O B-LOC I-LOC O O O O O B_Metric I_Metric I_Metric I_Metric O O O B_Thing I_Thing I_Thing I_Thing O B_Thing I_Thing I_Thing I_Thing O")

example_2 = InputExample(
    text = "工 商 货 币 基 金 会 发 布 通 知 ， 称 农 业 银 行 将 面 临 2 0 % 的 赤 字 。",
    label= "B-ORG I-ORG I-ORG I-ORG I-ORG I-ORG I-ORG O O B_Thing I_Thing O O B-ORG I-ORG I-ORG I-ORG O O O B_Metric I_Metric I_Metric O O O O"
)

example = example_1

check_point="/data/songjh/bert/finetuned/chi_literature/1/ner-2_6041.ckpt"

def get_label_list(file_path):
    with open(file_path, 'r') as file:
        # for every line read the labels in it.
        label_list = [line.strip() for line in file]
    return label_list


# Step 1: Apply a function to each column that finds the most frequent value
def most_frequent(arr):
    """Return the most frequent element in the array (the mode)."""
    counts = np.bincount(arr)  # Counts the occurrences of each value
    return np.argmax(counts)    # Returns the index (value) with the highest count

if __name__ == "__main__":

    label_list = get_label_list("/data/songjh/mindrecord/literature_NER/label_list.txt")
    
    example.label = example.label.replace("-ORG","_Organization")
    example.label = example.label.replace("-LOC","_Location")
    example.label = example.label.replace("-PER","_Person")

    max_seq_len=128
    vocab_file="/data/songjh/bert/vocab.txt"
    tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

    feature = convert_single_example(ex_index=0,example=example,
                           label_list=label_list,vocab_file=vocab_file,
                           mode="",output_dir="",
                           max_seq_length=max_seq_len, tokenizer=tokenizer)

    tag_to_index = convert_labels_to_index(label_list)
    
    # add extra <START> and <END> for CRF layer
    # teg_to_index is provide to CRF during training because CRF need to translate tag to index to record the path score.
    max_val = max(tag_to_index.values())
    tag_to_index["<START>"] = max_val + 1
    tag_to_index["<STOP>"] = max_val + 2
    number_labels = len(tag_to_index)
    label_list.append("<START>")
    label_list.append("<STOP>")

    # now make instance of model, BertNER is net with loss, 
    # which returns loss after CRF,
    # and BertNERModel will return logits emit-scores matrix, not final score after CRF 
    # bert_net_cfg = vars(bert_net_cfg)
    net_for_test = BertNER(bert_net_cfg, batch_size = 1, is_training = False, num_labels = number_labels, 
                                  with_lstm=True,use_crf=True,
                                  tag_to_index = tag_to_index)
    net_for_test.set_train(False)
    param_dict = load_checkpoint(check_point)
    load_param_into_net(net_for_test, param_dict)
    model = Model(net_for_test)

    # send data to GPU, because the batch size = 1 , so add an array.
    input_ids = Tensor([feature.input_ids], dtype=mstype.int32)
    input_mask = Tensor([feature.input_mask], dtype=mstype.int32)
    input_segment = Tensor([feature.segment_ids], dtype=mstype.int32)
    
    ori_text = example.text.split()
    ori_label = example.label.split()
    
    real_seq_length = len(ori_text) + 2
    real_seq_length_tensor = Tensor([real_seq_length],dtype=mstype.int32)

    logits = model.predict(input_ids, input_mask, input_segment, 
                        feature.label_ids, real_seq_length_tensor)

    #output result and compare it with ground truth
    # print(logits)
    logits = logits[0]
    best_pred = []
    for i in range(real_seq_length):
        pred_y = logits[i][0][0] # use CRF ,no need for argmax, directly pick out the index
        # use max voting
        pred_y =  most_frequent(pred_y)
        best_tag = label_list[pred_y]
        best_pred.append(best_tag)
    
    print("token truth predict")
    print(f"- <START> {best_pred[0]}")
    # print(f"- [CLS] {best_pred[1]}")
    for i in range(len(ori_text)):
        print(f"{ori_text[i]} {ori_label[i]} {best_pred[i+1]}")
    # print(f"- [SEP] {best_pred[len(ori_text)+2]}")
    print(f"- <STOP> {best_pred[len(ori_text)+1]}")