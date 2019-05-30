#!/bin/bash

timestamp=`date "+%d.%m.%Y_%H.%M.%S"`
output_dir='./logs/word_attention_non_chronological/'
config_file='./configs/word_attention_non_chronological.txt'

export CUDA_VISIBLE_DEVICES=1

mkdir -p $output_dir
#sudo touch ${output_dir}log.dev_${timestamp}.txt

echo ${config_file}
timestamp=`date "+%d.%m.%Y_%H.%M.%S"`
python3 -u main.py ${config_file} log.word_attention_non_chronological.txt 2>&1 | sudo tee ${output_dir}log.dev_${timestamp}.txt
