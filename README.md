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

之前 mindspore 安装 1.9 版本，只是因为 mindspore-hub 最高只有 1.9

其中可以下载的预训练、微调后模型权重，都在这个仓库列表中。但要注意这些 markdown “配置” 文件中，数据集的名称可能有大小写问题，例如 ChineseNER，应该改为 chieseNer。

https://gitee.com/mindspore/hub/tree/master/mshub_res/assets/mindspore/1.9

在 scripts/get_hub_ckpt.ipynb 中可以下载其中的模型权重，权重会下载到 
“/home/user_name/.mscache/mindspore/1.9/” 中。

之后便可以按照框架指示加载 bert 预训练/已微调模型权重。

#### 关于 mindnlp：

更方便的 NLP 套件，支持 mindspore 2.2 和 hugging face 数据集，但使用时发现权重载入由于命名问题会失败，如 Transformer，encoder layer的LayerNorm 参数，weight 和 bias 被写成 alpha 和 gamma。

对于小型模型比较难调节，从其文档的模型库
https://mindnlp.cqu.ai/supported_models/
可知，在微调 LLM 大模型时可能比较方便。

### 运行：

#### 准备数据集

在 src/generate_mindrecord 中，利用 generate_chinesener_mindrecord.bash ，可以将 指定位置的 BIO 类型的标注数据集（example.train, example.dev, example.test）转化为 mindrecord 类型的数据集，并存储在指定位置中。这里的 dev 相当于 validation set 。

要注意，若数据集中一句话长度超过设置的 max_len ，会直接被截断，若想提高利用率，可以在除了 “。”的其他标点处分隔。

NER 模型使用的数据集：



人民日报：https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily
中国文学（有更丰富的实体）：https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/tree/master?tab=readme-ov-file


#### 微调模型

在 scripts/run_ner_gpu.bash 中调整微调需要的各种参数。 (不要在 task_nere_config.yaml 中调整，会被覆盖)

使用之前在 data/mindrecord 准备的数据进行微调：

```bash
bash scripts/run_ner_gpu.sh 0
```

其中设置 do-train = true，do-eval = true ，可以选择微调参数，并且对 example.dev 数据集进行验证。

而 do_one_test.py 可以自定义模型测试样例，可以输入一个 sequence ，运行

```
python3 do_one_test.py
```

以查看载入指定 checkpoint 模型的预测结果是否符合预期。

但需要注意，可识别的实体 Label 限定于 label_list.txt， 如果要进行其他类别的分类，在载入预训练模型时，需要去除BERT 之后 BiLSTM-Dense1-CRF 这些 classification layer 的权重，以重新训练最后一层。

#### 导出和部署

如果 test 数据集和 do_one_test 结果都令人满意，则可以将模型封装到 docker 环境中，打包为镜像，最后部署到测试环境，并且提供 API 接口以完成后端的部署。
