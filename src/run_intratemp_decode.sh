#!/bin/bash

counter=1
while [ $counter -le 10 ]
do 
  echo $counter
  ((counter++))
  echo =======================train======================
  python run_summarization.py --mode=train --data_path=../data/finished_files/chunked/train_*.bin --vocab_path=../data/finished_files/vocab --log_root=log --input_attention=0 --attention_model=0 --pointer_gen=False --use_intra_decoder_attention=2 --batch_size=16 --lr=0.0001 --hidden_dim=200 --exp_name=IntraTemp_Decode_batch16_Mar18
  echo ==============decode on train dataset==============
  python run_summarization.py --mode=decode --data_path=../data/finished_files/chunked/train_000.bin --vocab_path=../data/finished_files/vocab --log_root=log --input_attention=0 --attention_model=0 --pointer_gen=False --use_intra_decoder_attention=2 --batch_size=16 --lr=0.0001 --hidden_dim=200 --exp_name=IntraTemp_Decode_batch16_Mar18 --single_pass=1
  echo ==============decode on val dataset=================
  python run_summarization.py --mode=decode --data_path=../data/finished_files/chunked/val_000.bin --vocab_path=../data/finished_files/vocab --log_root=log --input_attention=0 --attention_model=0 --pointer_gen=False --use_intra_decoder_attention=2 --batch_size=16 --lr=0.0001 --hidden_dim=200 --exp_name=IntraTemp_Decode_batch16_Mar18 --single_pass=1 
done

echo ALL DONE
