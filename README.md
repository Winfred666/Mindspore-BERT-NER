## Mindspore 1.9 BERT-LSTM-CRF 模型，NER实体识别任务 实践

来自校企联合-NLP趣味项目

整个项目来自 https://github.com/mindspore-ai/models/blob/r1.9/official/nlp/bert/README_CN.md，并有所精简，删除了 SQUAD 任务的 bert 训练脚本，以及在 Ascend 和 Model_arts 平台训练的脚本。

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

注意默认配套的 pillow 库版本错误，超过10，要降级。同时安装缺失的库：

```bash
pip uninstall pillow
pip install pillow==9.5.0
pip install decorator
```


之后运行验证程序，如果正确输出，则 mindspore 安装成功。

```bash
python3 -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"
```

#### mindspore-hub 配置：

之前 mindspore 安装 1.9 版本，只是因为 mindspore-hub 最高只支持 1.9

其中可以下载的预训练、微调后模型权重，都在这个仓库列表中：

https://gitee.com/mindspore/hub/tree/master/mshub_res/assets/mindspore/1.9

但要注意这些 markdown “配置” 文件中，数据集的名称可能有大小写问题，例如 ChineseNER，应该改为 chieseNer。

其实不安装 mindspore-hub 也没关系，直接去上述仓库，安装 markdown 中的链接，下载 ckpt 权重和模型代码文件即可。


在 scripts/get_hub_ckpt.ipynb 中可以下载其中的模型权重，权重会下载到 
“/home/\<user_name\>/.mscache/mindspore/1.9/” 中。

之后便可以按照框架指示加载 bert 预训练/已微调模型权重。

#### 关于 mindnlp：

更方便的 NLP 套件，支持 mindspore 2.2 和 hugging face 数据集，但使用时发现权重载入由于命名问题会失败，如 Transformer，encoder layer的LayerNorm 参数，weight 和 bias 被写成 alpha 和 gamma。

对于小型模型比较难调节，从其文档的模型库
https://mindnlp.cqu.ai/supported_models/
可知，在微调 LLM 大模型时可能比较方便。

### 运行：

基础思路是训练三个模型，以在对话中抽取样本，三个模型都是基于 Bert ：

1. 对话连贯分析，用于分割对话中的话题。使用的模型是 DuantiDCE，分析上下句是否连贯。

注意此处的任务并非分析语义是否相似（语义相似数据集一半十分严格，不适合分析对话连贯性），而是分析句子之间是否可能有逻辑连贯性。

由于 DuantiDCE 使用 pytorch ，转为 mindspore 框架时，Loss 部分代码编写有困难，因此保持在 pytorch 上进行中文语料的训练，最后转换模型权重 Checkpoint 为 mindspore 版本，以进行结果的预测。为了使用相同的配置， 单独安装 transformers 以使用 bert-base-uncased 或 bert-base-chinese 对应的 AutoTokenizer 。

> 根据输出信息，其实刚好够模型迁移：None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.

2. 对于分割出的一个段落，进行事件抽取。 事件抽取采用 Pipeline 模型。

参考：https://www.lixiaofei2yy.website/%E4%BA%8B%E4%BB%B6%E6%8A%BD%E5%8F%96%E5%AE%9E%E6%88%98

其第一部分，命名实体识别 NER 模型刚好使用 mindspore 框架提供的 Bert + BiLSTM + CRF ，配合微调好的 ChineseNer_bert_base 权重权重即可实现。

为了适应对话中可能出现的偏日常的实体（而不仅仅是国家、组织、人名），使用 [Chinese_literature](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/tree/master?tab=readme-ov-file) 数据集进行实体类别的扩充。

或者可以用更好的 ACE 2005 或 ERE 数据集，针对事件论元进行实体类别改良，但是要付费。

3. 最后是 Pipeline 第二部分，关系抽取 RE 模型。使用的模型是 

R-BERT：https://github.com/monologg/R-BERT

或其改进版本 EC-BERT：https://github.com/DongPoLI/EC-BERT

由于时间关系，最后并没有实现第三部分。

#### 准备数据集

在 src/generate_mindrecord 中，利用 generate_chinesener_mindrecord.bash ，可以将 指定位置的 BIO 类型的标注数据集（ 三个文件改名为 example.train, example.dev, example.test ）转化为 mindrecord 类型的数据集，并存储在指定位置中。这里的 dev 相当于 validation set 。

要注意，若数据集中一句话长度超过设置的 max_len ，会直接被截断，若想提高利用率，可以在除了 “。”的其他标点处分隔。


1. 对话语义相似度使用的数据集（其实就是机器人对话评估数据集）：



[中文语料库](https://github.com/codemayq/chinese-chatbot-corpus?tab=readme-ov-file) 中的数据集很全面，格式也很标准，但只有关于上下句的对话，适合训练 SQUAD，即聊天机器人，回答问题。 如果要作为评估对话连贯性的数据集，还需给每对问答打上连贯性分数。

而[DuantiDCE 模型](https://github.com/James-Yip/QuantiDCE) 则完全适合这个任务，对于评价对话连贯性采用了更加细化的打分方法，只不过语言为英文。

2. NER 模型数据集：

人民日报：

https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily

Chinese_literature（有更丰富的实体标签，但数据量更少）：

https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/tree/master?tab=readme-ov-file

3. RE 模型数据集：

[Chinese_literature](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/tree/master?tab=readme-ov-file) 还附带了一份关系抽取的数据集，刚好可以配套 NER 使用。


#### 微调模型

1. QuantiDCE

对于 QuantiDCE，使用其原始 pytorch 模型进行训练，最后在 安装有 pytorch 的 python 环境下，运行模型权重导出的脚本：

```bash
python ./quantiDCE/from_torch_ckpt.py
```

之后便可以将环境切换为装有 mindspore-gpu 的 python 环境，测试模型：

```bash
python one_test_quantiDCE.py
```

2. ChineseNER

在 scripts/run_ner_gpu.bash 中调整微调需要的各种参数。 (不要在 task_nere_config.yaml 中调整，会被覆盖)

使用之前在 data/mindrecord 准备的数据进行微调：

```bash
bash scripts/run_ner_gpu.sh 0
```

其中设置 do-train = true，do-eval = true ，可以选择微调参数，并且对 example.dev 数据集进行验证。

而 one_test_ner.py 可以自定义模型测试样例，可以输入一个 sequence ，运行之前微调好的模型，查看效果。

```bash
python one_test_ner.py
```

以查看载入指定 checkpoint 模型的预测结果是否符合预期。

但需要注意，可识别的实体 Label 限定于 label_list.txt， 如果要进行其他类别的分类，在载入预训练模型时，需要去除BERT 之后 BiLSTM-Dense1-CRF 这些 classification layer 的权重，以重新训练最后一层。


#### 导出和部署

由于 MINDIR 模型并不支持 CPU 部署，所以使用 ONNX。

对于quantiDCE 模型，在 one_test_quantiDCE.py 中，已经自带了 导出为 ONNX 的 步骤。

对于 ChineseNER 模型，运行下列命令输出 ONNX：

```bash
bash scripts/export.sh
```

