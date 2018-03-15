python run_summarization.py --mode=train --data_path=../data/finished_files/chunked/train_*.bin --vocab_path=../data/finished_files/vocab --log_root=log --input_attention=0 --attention_model=0 --pointer_gen=True --use_intra_decoder_attention=2 --batch_size=64 --exp_name=intra_pointer_Mar14

python run_summarization.py --mode=decode --single_pass=1 --data_path=../data/finished_files/chunked/val_*.bin --vocab_path=../data/finished_files/vocab --log_root=log --input_attention=0 --attention_model=0 --pointer_gen=True --use_intra_decoder_attention=2 --batch_size=64 --exp_name=intra_pointer_Mar14

python run_summarization.py --mode=decode --data_path=../data/finished_files/chunked/val_*.bin --vocab_path=../data/finished_files/vocab --log_root=log --input_attention=0 --attention_model=0 --pointer_gen=True --use_intra_decoder_attention=2 --batch_size=64 --exp_name=intra_pointer_Mar14

python run_summarization.py --mode=eval --data_path=../data/finished_files/chunked/val_*.bin --vocab_path=../data/finished_files/vocab --log_root=log --input_attention=0 --attention_model=0 --pointer_gen=True --use_intra_decoder_attention=2 --batch_size=64 --exp_name=intra_pointer_Mar14
