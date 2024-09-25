# just export the model to mindir/ONNX, but the latter is more popular (mindir suck, only for GPU)

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
python ${PROJECT_DIR}/../export.py \
    --description "run_ner" \
    --config_path "../../task_ner_config.yaml" \
    --export_ckpt_file "/data/songjh/bert/final_res/chineseNER/ner-2_6041.ckpt" \
    --label_file_path "/data/songjh/mindrecord/literature_NER/label_list.txt" \
    --export_file_name "/data/songjh/ONNX/chineseNER" \
    --file_format "ONNX" \
    --config_path="../../task_ner_config.yaml" \
    \
    \
    --do_train="false" \
    --do_eval="false" \
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
    --save_finetune_checkpoint_path="/data/songjh/bert/finetuned/chi_literature/3" \
    --load_finetune_checkpoint_path="/data/songjh/bert/finetuned/chi_literature/3" \
    \
    --load_pretrain_checkpoint_path="/data/songjh/pretrained//data/songjh/pretrained/bertfinetune_bilstmcrf_ascend_v190_chinesener_official_nlp_F1score96.07.ckpt" \
    \
    --label_file_path="/data/songjh/mindrecord/literature_NER/label_list.txt" \
    --train_data_file_path="/data/songjh/mindrecord/literature_NER/train.mind_record" \
    --eval_data_file_path="/data/songjh/mindrecord/literature_NER/eval.mind_record" \
    \

# --device_target "CPU" \