## Mindspore 1.9 BERT-LSTM-CRF 模型，NER实体识别任务 实践

来自校企联合-NLP趣味项目

整个项目来自 https://github.com/mindspore-ai/models/blob/r1.9/official/nlp/bert/README_CN.md，只保留了 mindrecord 数据生成、 NER 微调、模型导出部分的脚本。

### 准备工作

#### mindspore-gpu 1.9 配置：

mindspore 的环境配置比较麻烦，可以参考以下步骤：

1. 由于 mindspore GPU 版本只支持 Linux 平台；因此 Windows 环境需要在 WSL2 中安装 Cuda、CuDNN。

https://blog.csdn.net/luxun59/article/details/129642581

注意 Cuda Toolkit 版本必须为 11.6 或 11.1，以匹配 mindspore 要求，即使Nvidia驱动支持更高版本的 Cuda toolkit.

2. 接着是 gcc 版本， 必须为 10，不能过高，在 Linux可以通过更改软连接实现。

3. 然后是 python 版本，mindspore 最高支持 3.9 版本，同样不能过高。

4. 在安装完成后，才可以安装 mindspore-gpu：
https://www.mindspore.cn/versions#1.9.0

以 python3.9，Linux_x86_64，Cuda 11.1 为例：
```bash
pip3 install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.9.0-cp39-cp39-linux_x86_64.whl
```

#### mindspore-hub 配置：

之前 mindspore 安装 1.9 版本，只是因为 mindspore-hub 最高只有 1.9

其中可以下载的预训练、微调后模型权重，都在这个仓库列表中。但要注意这些 markdown “配置” 文件中，数据集的名称可能有大小写问题，例如 ChineseNER，应该改为 chieseNer。

https://gitee.com/mindspore/hub/tree/master/mshub_res/assets/mindspore/1.9

在 scripts/get_hub_ckpt.ipynb 中可以下载其中的模型权重，权重会下载到 
“/home/user_name/.mscache/mindspore/1.9/” 中。

之后便可以按照框架指示加载 bert 预训练/已微调模型权重。

#### 关于 mindnlp：

更方便的 NLP 套件，支持 mindspore 2.2 和 hugging face 数据集，但使用时发现权重载入由于命名问题会失败，如 Transformer，encoder layer的LayerNorm 参数，weight 和 bias 被写成 betwfdh1a 和 gamma。

对于小型模型比较难调节，从其文档的模型库
https://mindnlp.cqu.ai/supported_models/
可知，在微调 LLM 大模型时可能比较方便。

### 运行：

#### 准备数据集

#### 微调模型

在 scripts/run_ner_gpu.bash 中调整微调需要的各种参数。 (不要在 task_nere_config.yaml 中调整，会被覆盖)

使用之前在 data/mindrecord 准备的数据进行微调：

```bash
bash scripts/run_ner_gpu.sh 0
```


#### 导出和部署

