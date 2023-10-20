This project is related to one of the CVPR2020-retail vision challenges. The task of the challenge is to detect products in crowded store displays based on the SKU- 110K dataset.

# Challenge Overview : Product Detection in Densely Packed Scenes
<B> From the web</B> : [Visit](https://retailvisionworkshop.github.io/detection_challenge_2020/)

The world of retail takes the detection scenario to unexplored territories with millions of possible facets and hundreds of heavily crowded objects per image. This challenge is based on the [SKU-110K dataset](https://github.com/eg4000/SKU110K_CVPR19) collected from Trax’s data of supermarket shelves and pushes the limits of detection systems

<div align="center">
  <img src="https://github.com/kavindukalinga/Product-Detection-in-Densely-Packed-Scenes/blob/main/teaser.png" alt="A typical image in SKU-110K, showing densely packed objects">
  
  A typical image in SKU-110K, showing densely packed objects
</div>

### Dataset
The SKU-110K dataset collects 11,762 densely packed shelf images from thousands of supermarkets around the world, including locations in the United States, Europe, and East Asia. The dataset is provided solely for academic and non-commercial purposes.

<div align="center">
  <img src="https://github.com/kavindukalinga/Product-Detection-in-Densely-Packed-Scenes/blob/main/benchmarks_comparison.jpg" alt="Dataset Table">
  
</div>

Comparison of related benchmarks. #Img.: number of images. #Obj./img.: average items per image. #Cls.: number of object classes (more implies a harder detection problem due to greater appearance variations). #Cls./img.: average classes per image. dense: objects are typically densely packed. Idnt: images contain multiple identical objects or hard to separate object sub-regions. BB: bounding box labels are available.

# Our Solution
We used supervised learning..


## Coding
We used Jupyter Notebook provided by Google Colab (free version) to write our codes. We run our codes with GPU (Changed CPU to GPU/TPU).

<div align="center">
  <img src="https://github.com/kavindukalinga/Product-Detection-in-Densely-Packed-Scenes/blob/main/teaser.png" alt="Google Colab Jupyter Notebook">
  
  Figure: Our Coding Environment
</div>

We installed packages  YOLO, super-gradients, roboflow, supervision, opencv-python, fastapi, kaleido, python-multipart, uvicorn, tensorflow using below code.
```python
!pip install super-gradients roboflow supervision opencv-python fastapi kaleido python-multipart uvicorn tensorflow
!DEBIAN_FRONTEND=noninteractive apt update -y && apt install -y libglu1 libglib2.0-0 libsm6 libxrender1 libxext6 git build-essential
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### References

https://retailvisionworkshop.github.io/detection_challenge_2020/
https://retailvisionworkshop.github.io/cvpr2020/

https://github.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5

#### Kavindu Kalinga
<p align="left">
<a href="https://www.linkedin.com/in/kalingachandrasiri" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="kalingachandrasiri" height="15" width="20" /></a>
<a href="https://twitter.com/yuk_kalinga_c" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="kavindukalinga" height="15" width="20" /></a>
<a href="https://stackoverflow.com/users/16277941/kavindu-kalinga" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/stack-overflow.svg" alt="kavindu-kalinga" height="15" width="20" /></a>
<a href="https://www.facebook.com/kavindu.kalinga" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/facebook.svg" alt="kavindu.kalinga" height="15" width="20" /></a>
<a href="https://www.instagram.com/kavindu_kalinga" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="kavindu_kalinga" height="15" width="20" /></a>
<!-- <a href="https://www.youtube.com/c/uckvw2mrlhn_qxktjxyzahzw" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/youtube.svg" alt="uckvw2mrlhn_qxktjxyzahzw" height="15" width="20" /></a> -->
<a href="https://discord.gg/CrazzyHawK#8536" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/discord.svg" alt="CrazzyHawK#8536" height="15" width="20" /></a>
</p>


# More on the Challenge

## Challenge Info
This challenge includes a single track, where participants are invited to develop and train their methods using the data in the SKU-110K dataset and be tested on a yet to be released test set.

All the data in the SKU-110K dataset may be used for training, including the validation and test sets. Methods will be evaluated on a new test set that will be released The test set will be published without annotations. Detection results will be evaluated using the [Evaluation code](https://github.com/kavindukalinga/Product-Detection-in-Densely-Packed-Scenes/tree/main/Evaluation%20code).

### Challenge Results
1st place:  [Technical report](https://github.com/kavindukalinga/Product-Detection-in-Densely-Packed-Scenes/blob/main/1st_A%2BSolution%2Bfor%2BProduct%2BDetection%2Bin%2BDensely%2BPacked%2BScenes.pdf)

2nd place:  [Technical report](https://github.com/kavindukalinga/Product-Detection-in-Densely-Packed-Scenes/blob/main/2nd_Working_with_scale__2nd_place_solution_to_Product_Detection_in_Densely_Packed_Scenes.pdf)

## Overview : RetailVision - CVPR 2020
<div align="center">
<B> Revolutionizing the World of Retail </B>
  
<B> New Computer Vision Challenges </B>

<B> 06.15.2020 Afternoon </B>
</div>

<B> From the web</B> : [Visit](https://retailvisionworkshop.github.io/cvpr2020/)

The rapid development in computer vision and machine learning has caused a major disruption in the retail industry. In addition to the rise of the web and online shopping, traditional markets also quickly embrace AI-related technology solutions at the physical store level. Following the introduction of computer vision to the world of retail a new set challenges emerged, such as the detection of products in crowded store displays, fine-grained classification of many visually similar classes, as well as dynamically adapting to changes in data in terms of class appearance variation over time, and new classes that may appear in the images before they are labeled in the dataset. The scene complexity, scale, class imbalance, lack of reliable supervised samples, and dynamic nature of the data, encourage solutions such as context based detection and classification, few-shot learning, uncertainty modeling and open set recognition, and so forth.

This workshop aims to present and progress the revolution that is already occuring in the word of retail and welcomes any work on relevant computer vision challenges, including but not limited to:

- Detection in densely packed scenes
- Class imbalance and lack of labeled data. New classes introduced over time
- Ultrafine-grained object classification: Classes are often virtually indistinguishable by visual appearance
- Hierarchical classification: products fall into product, brand, and sub-brand hierarchies
- Context modeling of geometric structures
- Multi-person tracking
- Recognition of actions such as taking/returning/examining products
