# cbert_aug
This is a implementation of paper "Conditional BERT Contextual Augmentation" https://arxiv.org/pdf/1812.06705.pdf.
Our original implementation was two-stage, for convenience, we rewrite the code. 
The *global.config* contain the global configuration for bert and classifier.

To run the code, you should
1. python finetune_dataset.py
2. python aug_dataset.py

The hyperparameters of the models and training were selected by a grid-search using baseline models without data augmentation in each task’s validation set individually.


The classifier code is from <https://github.com/pfnet-research/contextual_augmentation>, thanks to the author.
