# Enhanced Attention-Based Multimodal Deep Learning for Product Categorization on E-Commerce Platform
This is the source-code of "Enhanced Attention-Based Multimodal Deep Learning for Product Categorization on E-Commerce Platform" paper published in "Lecture notes in Networks and Systems"

## Abstract
abeling and classifying a large number of products is one of the key challenges that e-commerce managers face. Building an automatic model that can accurately classify products helps to optimize the consumer search experience and ensure that they can easily find the products that meet their needs. In this study, we propose an improved Multimodal Deep Learning Model, based on the attention mechanism. This model has the ability to significantly improve accuracy over both traditional Unimodal Deep Learning and Multimodal Deep Learning models. The accuracy of our proposed model reaches 91.18% in classifying 16 different product categories. Meanwhile, traditional Multimodal Deep Learning models only achieved a modest accuracy of 77.21%. This result not only improves the searchability and online shopping experience of consumers, but also makes a significant contribution to solving the challenge of product classification on e-commerce platforms.

## Create checkpoint folder

Make directory to store the model's checkpoints 
~~~
mkdir model_checkpoint\checkpoint\version_1
~~~
Then putting the model's checkpoints into the folder

## Requirements ⚡️

~~~
conda env create -f environment.yaml
conda activate multimodal
~~~

## Run App.py ☀️

~~~
python App.py
~~~

## Run myapp 🌗

~~~
cd myapp
npm run dev
~~~

## Paper citation
~~~
@inproceedings{hung2024enhanced,
  title={Enhanced attention-based multimodal deep learning for product categorization on e-commerce platform},
  author={Hung, Le Viet and Binh, Phan and Minh Nhat, Phan and Van Hieu, Nguyen},
  booktitle={Conference on Information Technology and its Applications},
  pages={87--98},
  year={2024},
  organization={Springer}
}
~~~
