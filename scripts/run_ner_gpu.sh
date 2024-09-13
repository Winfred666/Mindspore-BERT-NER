#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_ner_gpu.sh DEVICE_ID"
echo "DEVICE_ID is optional, default value is zero"
echo "for example: bash scripts/run_ner_gpu.sh 1"
echo "assessment_method include: [BF1, MF1, clue_benchmark]"
echo "=============================================================================================================="

if [ -z $1 ]
then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES="$1"
fi

mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

python3 ${PROJECT_DIR}/../run_ner.py  \
    --config_path="../../task_ner_config.yaml" \
    --device_target="GPU" \
    \
    --do_train="true" \
    --do_eval="true" \
    \
    --assessment_method="BF1" \
    --use_crf="true" \
    --with_lstm="true" \
    --epoch_num=5 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="true" \
    --train_batch_size=4 \
    --eval_batch_size=2 \
    \
    --vocab_file_path="/data/songjh/bert/vocab.txt" \
    --save_finetune_checkpoint_path="/data/songjh/bert/finetuned/" \
    --load_finetune_checkpoint_path="/data/songjh/bert/finetuned/" \
    \
    --load_pretrain_checkpoint_path="/data/songjh/bertfinetune_bilstmcrf_ascend_v190_chinesener_official_nlp_F1score96.07.ckpt" \
    \
    --label_file_path="/data/songjh/mindrecord/literature_NER/label_list.txt" \
    --train_data_file_path="/data/songjh/mindrecord/literature_NER/train.mind_record" \
    --eval_data_file_path="/data/songjh/mindrecord/literature_NER/dev.mind_record" \
    \
    --schema_file_path=""
# > ner_log.txt 2>&1 &
# do not run the command in the background , and log into file, need to see error directly 