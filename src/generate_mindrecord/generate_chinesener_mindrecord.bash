# WARNING: only the labels in --labels parameter will be generate as Named entity. Others that not appear will be non-entity 'O'.

python3 generate_chinesener_mindrecord.py --output_dir="/data/songjh/mindrecord/literature_NER" \
    --vocab_file="/data/songjh/bert/vocab.txt" \
    --max_seq_length=128 \
    --data_dir="/data/songjh/raw_dataset/literature_NER_dataset" \
    --labels="O B_Location I_Location B_Organization I_Organization B_Thing I_Thing B_Person I_Person B_Time I_Time B_Metric I_Metric"
