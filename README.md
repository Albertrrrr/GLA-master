## TDCL
This project contains the main source code of TDCL. The model within it is sourced from Huggingface, and you can download and use them directly. This source code is also inspired by "Supporting Clustering with Contrastive Learning (NAACL 2021)", and the link is https://github.com/amazon-science/sccl.

### Getting Start
#### Dependencies:
```
python==3.6.13 
pytorch==1.6.0. 
sentence-transformers==2.0.0. 
transformers==4.8.1. 
tensorboardX==2.4.1
pandas==1.1.5
sklearn==0.24.1
numpy==1.19.5
```

#### Datasets
TDCL Download the original datastes from https://github.com/rashadulrakib/short-text-clustering-enhancement/tree/master/data

More importantly, you can use the AugData directly.
```
python main.py  
    --objective TDCL 
    --bert v2 
    --augtype virtual 
    --eta 10 
    --batch_size 64 
    --max_iter 7000 
    --num_classes 8 
    --time ISCCL-0728-SS-1 
    --dataname search_snippets_charswap_20 
```