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
## Trial Run
```
cd src
sh run_cmd.sh
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

### Baseline
Use PyTeaser generate the Summary From CNN https://github.com/xiaoxu193/PyTeaser 
save the summary. 
```
sumy_eval lex-rank ref_summ.txt --url=https://www.huffingtonpost.com/2013/11/22/twitter-forward-secrecy_n_4326599.html
```
sample output
```
Precision: 0.000000
Recall: 0.000000
F-score: 0.000000
Cosine similarity: 0.350201
Cosine similarity (document): 0.670010
Unit overlap: 0.128788
Unit overlap (document): 0.294798
Rouge-1: 0.161616
Rouge-2: 0.000000
Rouge-L (Sentence Level): 0.066107
Rouge-L (Summary Level): 0.039251
```
```
sumy_eval lex-rank art2_sum.txt --file=/Users/peli/forgit/nlp_project/data/art2.txt --format=plaintext
Precision: 0.000000
Recall: 0.000000
F-score: 0.000000
Cosine similarity: 0.721001
Cosine similarity (document): 0.900767
Unit overlap: 0.216450
Unit overlap (document): 0.344426
Rouge-1: 0.597403
Rouge-2: 0.461538
Rouge-L (Sentence Level): 0.103124
Rouge-L (Summary Level): 0.004133
```
### To-do List:
Intra-Temporal Attention Decoder on Input Sequence
Intra-Decoder Attention Decoder
Token Generation and Pointer
Sharing Decoder Weights

### Group sharing resources
Mendeley https://www.mendeley.com/newsfeed/#group:d48f9375-8628-3922-aa04-ac0f8832ae01
Slack https://cs224nprojectteam.slack.com

### Software Versions
pip install tensorflow==1.4.1
python 3.5.2 (or python 3.6.x)
