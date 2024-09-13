python3 generate_chinesener_mindrecord.py --output_dir="/data/songjh/mindrecord/literature_NER" \
    --vocab_file="/data/songjh/bert/vocab.txt" \
    --max_seq_length=128 \
    --data_dir="/data/songjh/raw_dataset/literature_NER_dataset" \
    --labels="I_Location I_Organization I_Thing B_Location B_Persion I_Person X B_Organization O B_Thing B_Time I_Time B_Metric I_Metric B_Abstract I_Abstract"