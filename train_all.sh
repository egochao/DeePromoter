#!/bin/bash
echo "This will perform training all experiments and save the results to ./output"

python train.py -d data/human/nonTATA/hs_pos_nonTATA.txt -e human_nonTATA
python train.py -d data/human/TATA/hs_pos_TATA.txt -e human_TATA

python train.py -d data/mouse/nonTATA/mm_pos_nonTATA.txt -e mouse_nonTATA
python train.py -d data/human/TATA/mm_pos_TATA.txt -e mouse_TATA