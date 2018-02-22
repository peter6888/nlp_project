# nlp_project
The final project for CS224n

The attempt to replicate below paper.

https://arxiv.org/abs/1705.04304

## Download data
```
cd data
sh download.sh
unzip finished_files.zip
```
## Train base model from https://github.com/abisee/pointer-generator
#### Pre-requirement, make sure have pyrouge installed
```
pip install pyrouge
```
#### Train
```
python run_summarization.py --mode=train --data_path=../nlp_project/data/finished_files/chunked/train_*.bin --vocab_path=../nlp_project/data/finished_files/vocab --log_root=/home/stonepeter/log --exp_name=baseline
```

#### Validate
```
python run_summarization.py --mode=eval --data_path=../nlp_project/data/finished_files/chunked/train_* --vocab_path=../nlp_project/data/finished_files/vocab --log_root=/home/stonepeter/log --exp_name=baseline
```
#### Beam Search Validate
```
python run_summarization.py --mode=decode --data_path=../nlp_project/data/finished_files/chunked/train_* --vocab_path=../nlp_project/data/finished_files/vocab --log_root=/home/stonepeter/log --exp_name=baseline
```
#### Result Example
> [data/sample_summary.txt](https://github.com/peter6888/nlp_project/blob/master/data/sample_summary.txt)
