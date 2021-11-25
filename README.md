# Local Discriminative Representation Learning
Pytorch code for Unsupervised Local Discrimination for Medical Images. [(arXiv)](https://arxiv.org/abs/2108.09440)
It is an extension of our previous work acceptted by IPMI 2021. [(Unsupervised Learning of Local Discriminative Representation for Medical Images)](https://link.springer.com/chapter/10.1007/978-3-030-78191-0_29).  
# Highlights
The goal of this work is to learn local discriminative representation for medical images. Medical images of human contain similar anatomical structures, and thus pixels can be classifying into several clusters based on their context. There are mainly two highlights:
1. Based on the priori knowledge that medical images of human share similar anatomical structures, we propose a unsupervised deep learning framework to learn discriminative features and cluster similar regions. In this framewor, two branch, including an embedding branch to embed each pixel and a clustering branch to cluster similar regions. In the embedding space, pixels of similar structures should be closely distributed. The learnt representation can be a good initialization for corresponding down-streams.
![](https://github.com/HuaiChen-1994/LDLearning/blob/main/figures/learning_region_discrimination.png)
2. Similar topological priors are shared among different medical images, and it is easy for specialists to identify target anatomical structures based on corresponding prior knowledge, including relative location, topological structure. Based on it, we add these priors to guide cluster branch to cluster specific regions.
![](https://github.com/HuaiChen-1994/LDLearning/blob/main/figures/prior_knowledge.png) 
