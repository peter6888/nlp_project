# nlp_project
The final project for CS224n

The attempt to replicate below paper.

https://arxiv.org/abs/1705.04304

## Download data
cd data
sh download.sh
unzip finished_files.zip

## Train base model from https://github.com/abisee/pointer-generator
pip install pyrouge
python run_summarization.py --mode=train --data_path=../nlp_project/data/finished_files/chunked/train_*.bin --vocab_path=../nlp_project/data/finished_files/vocab --log_root=/tmp/log --exp_name=baseline
