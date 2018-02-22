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
'''
INFO:tensorflow:ARTICLE:  -lrb- cnn -rrb- -- former u.s. surgeon general __joycelyn__ elders told cnn sunday she supports legalizing marijuana . the __trend-setting__ state of california is voting next month on a ballot initiative to legalize pot , also known as proposition 19 . the measure would legalize recreational use in the state , though federal officials have said they would continue to enforce drug laws in california if the initiative is approved . `` what i think is horrible about all of this , is that we criminalize young people . and we use so many of our excellent resources ... for things that are n't really causing any problems , '' said elders . `` it 's not a toxic substance . '' supporters of california 's __prop.__ 19 say it would raise revenue and cut the cost of enforcement , while opponents point to drug 's harmful side-effects . u.s. attorney general eric holder said in a letter , obtained by cnn friday , that federal agents would continue to enforce federal marijuana laws and warned __prop.__ 19 , if passed , would be a major stumbling block to federal partnerships between state and local authorities around drug enforcement . his letter was a response to an august letter from several former directors of the u.s. drug enforcement administration urging the white house to block __prop.__ 19 if it 's approved next month . elders stressed the drug is not physically addictive and pointed to the damaging impact of alcohol , which is legal . `` we have the highest number of people in the world being criminalized , many for non-violent crimes related to marijuana , '' said elders . `` we can use our resources so much better . ''
INFO:tensorflow:REFERENCE SUMMARY: __joycelyn__ elders tells cnn resources can be better spent . she says the drug 's !!__illegality__!! is !!__criminalizing__!! young people . `` it 's not a toxic substance , '' she says . california 's proposition 19 would legalize marijuana use in the state .
INFO:tensorflow:GENERATED SUMMARY: former u.s. surgeon general joycelyn elders told cnn sunday she supports legalizing marijuana . the measure would legalize recreational use in the state , though federal officials have said they would continue to enforce drug laws in california if the initiative is approved .
'''
