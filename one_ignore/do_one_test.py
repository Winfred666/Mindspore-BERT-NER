import mindspore as ms
from src.generate_mindrecord.generate_chinesener_mindrecord import convert_single_example,InputExample
import src.tokenization as tokenization
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.utils import convert_labels_to_index
from src.model_utils.config import bert_net_cfg
from src.bert_for_finetune import BertNER
from src.bert_model import BertConfig

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

import numpy as np

example = InputExample(
    text ="北 京 大 学 的 张 三 同 学 在 上 海 旅 游",
    label="B-ORG I-ORG I-ORG I-ORG O B-PER I-PER O O O B-LOC I-LOC O O")


check_point="/data/songjh/bert/finetuned/ner_1-8_25329.ckpt"


def get_label_list(file_path):
    with open(file_path, 'r') as file:
        # for every line read the labels in it.
        label_list = [line.strip() for line in file]
    return label_list

if __name__ == "__main__":
    label_list = get_label_list("/data/songjh/bert/mindrecord/label_list.txt")
    
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
    max_val = max(tag_to_index.values())
    tag_to_index["<START>"] = max_val + 1
    tag_to_index["<STOP>"] = max_val + 2
    number_labels = len(tag_to_index)
    label_list.append("<START>")
    label_list.append("<STOP>")

    # now make instance of model.
    # bert_net_cfg = vars(bert_net_cfg)
    net_for_test = BertNER(bert_net_cfg, 1, False, number_labels, with_lstm=False,
                                  use_crf=True, tag_to_index=tag_to_index)
    net_for_test.set_train(False)
    param_dict = load_checkpoint(check_point)
    load_param_into_net(net_for_test, param_dict)
    model = Model(net_for_test)

    # send data to GPU, because the batch size = 1 , so add an array.
    input_ids = Tensor([feature.input_ids], dtype=mstype.int32)
    input_mask = Tensor([feature.input_mask], dtype=mstype.int32)
    input_segment = Tensor([feature.segment_ids], dtype=mstype.int32)
    logits = model.predict(input_ids, input_mask, input_segment, feature.label_ids)

    #output result and compare it with ground truth
    logits = logits[0]
    best_pred = []
    ori_text = example.text.split()
    ori_label = example.label.split()
    for i in range(len(ori_text) + 4):
        pred_y = logits[i][0][0][0] # use CRF ,no need for argmax, directly pick out the index
        # best_tag = np.argmax(token,axis=0) 
        best_tag = label_list[pred_y]
        best_pred.append(best_tag)

    print("token truth predict")
    print(f"- <START> {best_pred[0]}")
    print(f"- [CLS] {best_pred[1]}")
    for i in range(len(ori_text)):
        print(f"{ori_text[i]} {ori_label[i]} {best_pred[i+2]}")
    print(f"- [SEP] {best_pred[len(ori_text)+2]}")
    print(f"- <STOP> {best_pred[len(ori_text)+3]}")