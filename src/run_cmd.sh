#!/usr/bin/env bash
python run_summarization.py --mode=train --data_path=../data/finished_files/chunked/train_*.bin --vocab_path=../data/finished_files/vocab --log_root=~/log --exp_name=baseline
