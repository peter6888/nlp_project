#!/usr/bin/env bash
python run_summarization.py --mode=train --data_path=../data/finished_files/chunked/train_*.bin --vocab_path=../data/finished_files/vocab --batch_size=4 --max_dec_steps=10 --attention_model=0 --use_intra_decoder_attention=2 --log_root=log --exp_name=ta_debug

