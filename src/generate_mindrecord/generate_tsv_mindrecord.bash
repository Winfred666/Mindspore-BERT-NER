# Here we can use multiple file and combine them to create one dataset.
# warning: for classification there are label list, 
# but for scoring task like "Quantifiable Dialogue Coherence Evaluation", 
# only ground truth y value is provided.
python3 generate_tnews_mindreocrd.py \
    --data_dir="/data/songjh/raw_dataset/clean_chat_corpus/chatterbot.tsv" \
    --task_name="chatterbot" \
    --vocab_file="/data/songjh/bert/vocab.txt" \
    --output_dir="/data/songjh/mindrecord/chatterbot" \
    --max_seq_length=128 \
    --manually_split=True

    