# conda activate torch
import torch

import pickle

torch_ckpt_path = "/home/songjh/QuantiDCE/output/71/bert_metric_kd_finetune/model_best_kd_finetune_loss.ckpt"
dict_ckpt_path = "/data/songjh/bert/finetuned/quantiDCE/chinese_finetuned.pkl"
# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典

def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        print(name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params



if __name__ == "__main__":
    param_dict = pytorch_params(torch_ckpt_path)
    # print all key of torch params.
    # param_names = param_dict.keys()
    # print(param_names)
    with open(dict_ckpt_path, 'wb') as f:
        pickle.dump(param_dict, f)