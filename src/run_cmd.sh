#!/usr/bin/env bash
python run_summarization.py --mode=train --data_path=../../../nlp_project/data/finished_files/chunked/train_*.bin --vocab_path=../../../nlp_project/data/finished_files/vocab --log_root=/Users/peli/log --exp_name=baseline
