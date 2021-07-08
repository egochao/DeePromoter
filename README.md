# DeePromoter
Pytorch implementation of [DeePromoter](https://doi.org/10.3389/fgene.2019.00286)
Active sequence detection for promoter(DNA subsequence regulates transcription initiation of the gene by controlling the binding of RNA polymerase)

<p align="center">
    <img src="figs/model.jpg" width="80%">
</p>

# Updates

- 2021-07-08 : Finish training and testing scripts for DeePromoter

# Training 
## Requirements 

- Please install torch==1.9 from https://pytorch.org

- You can install others Python dependencies with

    ```bash
    pip3 install -r requirements.txt
    ```

## Dataset 
Current supported dataset is:
- [EPDnew](https://epd.epfl.ch//index.php) : A collection of experimentally validated promoters for selected model organisms. Evidence comes from TSS-mapping from high-throughput expreriments such as CAGE and Oligocapping

## Preprocessing 

Dataset for Human and Mouse had been processed and stored in ./data

Procedure for create negative dataset as described in paper:

+ Step 1: Break the protein sequence to N part(20 as in the paper)

+ Step 2: Random choose M part of the original protein to keep it, and random initialize the rest

+ Step 3: For every training step mix the positive batch with negative batch and perform training

<p align="center">
    <img src="figs/negative_generation.jpg" width="80%">
</p>

##Training 
Train your model with. 

```
python train.py -d data/human/nonTATA/hs_pos_nonTATA.txt --experiment_name human_nonTATA
``` 

