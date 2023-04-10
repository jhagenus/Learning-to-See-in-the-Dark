---
title: 'Blog post'
disqus: hackmd
---

Group 7 - Reproduction of the paper 'Learning to See in the Dark'
===

Jeroen Hagenus, 5099528, j.hagenus@student.tudelft.nl
Kane de Roodt, 5520126, k.deroodt@student.tudelft.nl
Daan Scherrenburg, 5175151, d.j.scherrenburg@student.tudelft.nl

## Table of Contents
* [Sources](#head1) 
* [Introduction](#head2) 
* [Theory](#head3) 
* [Method](#head4) 
    * [Deep learning framework](#head41)
    * [Network architecture](#head42)
    * [Experiments](#head43)
    * [Training procedure](#head44)
* [Results](#head5) 
* [Conclusion and Discussion](#head6) 
* [References](#head7) 
* [Appendix and FAQ](#head8) 

## <a id="head1"></a>Sources
**Paper**: https://cchen156.github.io/paper/18CVPR_SID.pdf

**Data**: https://github.com/cchen156/Learning-to-See-in-the-Dark

**Github**: https://github.com/cchen156/Learning-to-See-in-the-Dark

**Paper website**: https://cchen156.github.io/SID.html

**Our Github**: https://github.com/jhagenus/ProjectDeepLearning


## <a id="head2"></a>Introduction
In this blogpost we will reproduce the “Learning to See in the Dark” paper by translating the code to PyTorch. We will also make some changes by making the code more readable and adding batch normalization to the used network. The paper 'Learning to See in the Dark' is written by Chen Chen, Qifeng Chen, Jia Xy and Vladlen Koltun and published in May of 2018. The paper presents a convolutional neural network that restores dark images into images that can be visually perceived. This is achieved by using two images, a long exposed image containing a lot of light functioning as a ground truth image and a short exposed low-light image serving as input data. 

Both sets of images were captured using a Sony a7S II on the same location and angle. The researchers were able to achieve a performance of 28.88 for the peak signal-to-noise ratio and 0.787 for structural similarity by feeding these images into the model.

Our blogpost will start by explaining the theory presented in the original paper to get a better understanding of the underlying processes. Afterward, in the 'method' chapter, we will first explain the change from TensorFlow to PyTorch followed by a section about the network architecture explaining the addition of Batch Normalization. We will then introduce the three different experiments and the training procedure. We will present the results of the experiments afterwards and end with a discussion and conclusion.


## <a id="head3"></a>Theory
The authors of the paper introduced the See-in-the-Dark (SID) dataset, specifically designed for training and testing low-light single-image processing techniques. The dataset comprises 5094 raw short-exposure images and 424 distinct long-exposure reference images, which cater to both indoor and outdoor environments. Two different cameras were utilized for capturing the images: the Sony α7S II and Fujifilm X-T2. These cameras have distinct sensors - the Sony has a full-frame Bayer sensor, while the Fuji has an APS-C X-Trans sensor. For the reproduction of this paper, only the Sony dataset will be used to reduce the number of processed images and decrease computation time. The Sony dataset comprises 2697 raw short-exposure images and 231 distinct long-exposure reference images. This simplification also aims to decrease the number of Python scripts required for reproduction, as both Sony and Fuji have distinct implementations for both the test and training scripts. To create a representative dataset, 20% of the images are assigned to the test set, and an additional 10% to the validation set.

Traditional image processing pipelines, such as the ones presented by Jiang et al. [[1]], use a sequence of modules to process raw data from imaging sensors. These pipelines, however, struggle to handle fast low-light imaging due to the extremely low signal-to-noise ratio (SNR). Hasinoff et al. [[2]] introduced a burst imaging pipeline that can produce better results, but it adds complexity and may not easily extend to video capture.

![pipeline](https://i.imgur.com/YUgZERx.png)

To address these limitations, the authors propose an end-to-end learning approach for direct single-image processing of fast low-light images. They train a fully-convolutional network (FCN) to perform the entire image processing pipeline, operating on raw sensor data. The pipeline structure is illustrated in the figure above. Input data is packed into multiple channels, this input data then enters the fully-convolutional network (presented as 'ConvNet' in the figure), and a sub-pixel layer recovers the original resolution from the half-sized output.

The researchers tested two general structures for the core FCN: a multi-scale context aggregation network (CAN) and a U-net. They opted for the U-net due to its memory efficiency, enabling the processing of full-resolution images in GPU memory. The amplification ratio determines the output brightness and can be adjusted by the user, similar to the ISO setting in cameras. The pipeline performs blind noise suppression and color transformation, outputting the processed image directly in the sRGB space. 

To validate the results of the processed images, two assessment metrics were employed: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM). PSNR represents the relationship between a signal's maximum potential power and the corrupting noise that influences its accuracy. Meanwhile, SSIM quantifies the resemblance between a pair of images. A higher value for both these parameters indicates a closer approximation of the reconstructed image to the source. The paper presents the following average PSNR and SSIM results for the default pipeline, which will be reproduces in this study with PyTorch: PSNR = 28.88, SSIM = 0.787.


## <a id="head4"></a>Method
In this section, we will explain the changes that we made on the original model for the reproduction and what experiments we executed to validate whether the changes should be kept. 

### <a id="head41"></a>Deep learning framework
The first major change to the original code is the deep learning framework. We decided to change the deep learning framework from TensorFlow version 1 to PyTorch. Both TensorFlow and PyTorch are popular frameworks for deep learning projects. However, they have slightly different use cases due to their different functionality.
The main reason for switching to PyTorch is that this framework is more intuitive to most developers. PyTorch is very well integrated in the Python language, this is the reason that using PyTorch feels more familiar. TensorFlow has a steeper learning curve, because most developers need to get familiar with this framework first. Because of this more intuitive feel of PyTorch, it is the preferred framework for developing rapid prototypes and research projects. [[3]]


### <a id="head42"></a>Network architecture
As explained in the ‘Theory’ section of this blogpost, the original code makes use of a fully-convolutional network (FCN) to perform the entire image processing pipeline. We decided to maintain most of this network the same as the writers of this original code probably performed a lot of validation in defining this network. However, we wanted to try to change something to the original model to see whether we could improve the results or to validate that this change should not be applied. We decided to apply Batch Normalization to the fully-convolutional network. 

Batch Normalization (BN) has been a proven method to improve certain methods. BN has been introduced in the paper ‘Batch normalization: Accelerating deep network training by reducing internal covariate shift’ by S. Ioffe and C. Szegedy.[[4]] This paper has, according to Google Scholar, been cited over 45000 times since then. BN can have several benefits. The first benefit is that it could speed up training of the model. This is because BN makes sure that the input has a similar distribution for every layer. Besides, BN can act as a form of regularization that can result in the model being less likely to overfit. Finally, BN can improve the performance of the model. BN can help to reduce the impact of noise in the image, resulting in a better reconstruction and therefore a better PSNR score. 

BN is usually implemented after every layer, in our case these are convolutional layers. However the original network structure makes use of double convolutional layers. This means that the input will go through two convolutional layers before applying a max pooling layer. Therefore, we need to determine whether it is better to apply BN just at the end of the double convolutional layer or after every single layer. 

### <a id="head43"></a>Experiments
In order to validate whether the applied changes result in a better performing model or not, some experiments need to be conducted. We came up with 3 experiments all using a different model.

The first experiment will be using the original code without any changes to the network layout, however it is transformed to the PyTorch framework. This experiment will validate whether the network will have a similar performance using the Pytorch framework compared to TensorFlow. 

The second experiment will be using a slightly modified network using Batch Normalization. In the section ‘Network architecture’ we already explained that the network consists of double convolutional layers. This experiment will validate whether applying BN after a double layer will result in a better performance. 

The third experiment will be applying BN after every convolutional layer. This will validate whether this results in a better performance compared to the original model and the model with a single batch normalization layer.

All experiments have been applied to the entire ‘Sony’ dataset that was used by the original paper. We decided to decrease the number of epochs to 1000 compared to the 4000 epochs of the original paper. The reason for this decrease was the training time. Training all three models on all data for 1000 epochs already took over 17 hours, so we simply did not have enough time and credits to run the models for all 4000 epochs. We will try to make a fair comparison using these results. 


### <a id="head44"></a>Training procedure
To train our model, we first started by running the code locally on our own laptops. The code includes a built-in debug feature to ensure that the model runs smoothly. Numerous variables play a role in the training process, including batch size, total number of epochs, and dataset size. In PyTorch users can choose to use either CPU or CUDA cores. For our training, we utilized CUDA cores because this will significantly reduce the training time over using CPU cores. However, for model testing, we used CPU cores not because this is faster but testing all images required more than 64GB of RAM. To Train the final model, we utilized Google Cloud with a Nvidia T4 GPU with 16GB of RAM, a 16-core CPU and 104GB of RAM. The original paper used 4000 epochs to train the model. But due to budget constraints (limited to $150 of Google Cloud storage) and time considerations, we opted to set the total number of epochs for the final three models to 1000. The size of the input dataset is another parameter that affects the training process. The final model was trained on 161 images the remaining of the images in the dataset were used for validation and testing purposes. 


## <a id="head5"></a>Results

In this section, we present the results of our experiments comparing the performance of the three PyTorch models with the TensorFlow baseline. The performance metrics used to evaluate the models are PSNR and SSIM. Higher PSNR and SSIM values indicate better image quality and structural similarity, respectively.

The PyTorch baseline model, which utilized the original network layout without any changes, was trained for 1000 epochs. This model achieved a PSNR of 28.13 and an SSIM of 0.755. The results show that the performance of the PyTorch model is relatively close to the TensorFlow baseline despite being trained for fewer epochs.

The second PyTorch model, with a single batch normalization layer added after each double convolutional layer, was also trained for 1000 epochs. This model yielded a PSNR of 25.51 and an SSIM of 0.700. Surprisingly, the performance of this model was worse than both the TensorFlow and PyTorch baselines, indicating that the addition of a single batch normalization layer did not improve the model's performance.

The third PyTorch model, which applied batch normalization after every convolutional layer, was trained for 1000 epochs as well. This model resulted in a PSNR of 25.30 and an SSIM of 0.686, again underperforming compared to the TensorFlow and PyTorch baseline models. This outcome suggests that applying batch normalization after every convolutional layer does not necessarily lead to better performance by using the metrics from the original paper.



| Model                       | # Epochs | PSNR  | SSIM  |
|-----------------------------|----------|-------|-------|
| Tensorflow baseline | 4000     | 28.88 | 0.787 |
| PyTorch baseline    | 1000     | 28.13 | 0.755 |
| PyTorch single batch norm   | 1000     | 25.51 | 0.700 |
| PyTorch double batch norm   | 1000     | 25.30 | 0.686 |

In addition to the quantitative results, we also provide a qualitative analysis of the images generated by the different models. Four representative images per model have been selected for comparison. These images offer a visual perspective on the performance of each model, allowing us to assess the image quality and structural similarity.

| <div style="width:100px"></div> Ground Truth | <div style="width:100px"></div> PyTorch baseline | <div style="width:100px"></div> PyTorch single batch norm | <div style="width:100px"></div> PyTorch double batch norm |
| -----| ----- | ----- | ---- |
|<img src="https://i.imgur.com/li774f9.jpg" alt= “image” width="1000"/>| <img src="https://i.imgur.com/tNVvMIF.jpg" alt= “image” width="1000" /> | <img src="https://i.imgur.com/yeN5TFl.jpg" alt= “image” width="1000" /> | <img src="https://i.imgur.com/AcFFVmD.jpg" alt= “image” width="1000" /> |
| <img src="https://i.imgur.com/xfZ63OA.jpg" alt= “image” width="1000" /> | <img src="https://i.imgur.com/Ta0eyho.jpg" alt= “image” width="1000" /> | <img src="https://i.imgur.com/dmUJftB.jpg" alt= “image” width="1000" />| <img src="https://i.imgur.com/j5XLFvG.jpg" alt= “image” width="1000" />|
| <img src="https://i.imgur.com/4i0Wbu1.jpg" alt= “image” width="1000" /> | <img src="https://i.imgur.com/ATDlYmB.jpg" alt= “image” width="1000" />| <img src="https://i.imgur.com/nvwxHTH.jpg" alt= “image” width="1000" />| <img src="https://i.imgur.com/J9uuIzP.jpg" alt= “image” width="1000" />|
| <img src="https://i.imgur.com/gHGECRv.jpg" alt= “image” width="1000" /> | <img src="https://i.imgur.com/zyppokA.jpg" alt= “image” width="1000" />| <img src="https://i.imgur.com/0ltoVj2.jpg" alt= “image” width="1000" />| <img src="https://i.imgur.com/XqLRzWx.jpg" alt= “image” width="1000" />|


## <a id="head6"></a>Discussion & Conclusion 
To validate if the translation from TensorFlow to PyTorch was successful we can look at the quantitative results. When comparing the TensorFlow baseline with the PyTorch baseline, we see that both PSNR and SSIM differ. The reason for the differences can be attributed to various factors, with the main reason being that our model was trained on 1000 epochs instead of the original model's 4000. We assumed that training the model on more epochs would result in more similar values, indicating a successful translation.

When we compare the experiments we conducted on the model to the PyTorch baseline, we can see that both the single batch normalization layer and the double batch normalization layer perform less well in the quantitative results. Both models score significantly lower in both PSNR and SSIM scores. This was unexpected, as batch normalization is supposed to help reduce the impact of noise in the photo, ultimately leading to better reconstruction.

When we compare the photos in the qualitative analysis table, starting with the first row where we see a fence of a basketball court with a grass field in front, we can see several interesting things. The photos with batch normalization contain more color, but when we zoom in on the photo, it contains less detail, making the fence blurrier and less visible. An interesting detail that stands out in the photo with double batch normalization is that the curb has yellow imperfections.

Looking at the second row of photos, we can also see that batch normalization makes the photo much lighter and the details less sharp. This trend continues in the remaining images, where it is also noticeable that batch normalization leads to more imperfections that are clearly visible in the third row of double batch normalization, where the bottle contains red dots that are not visible in the ground truth.

From the quantative and qualitative analysis, we can conclude that batch normalization does not improve photo reconstruction as previously assumed based on the literature research we conducted.


## <a id="head7"></a>References

[[1]] H. Jiang, Q. Tian, J. E. Farrell, and B. A. Wandell. Learning the image processing pipeline. IEEE Transactions on Image Processing, 26(10), 2017.

[[2]] S. W. Hasinoff, D. Sharlet, R. Geiss, A. Adams, J. T. Barron, F. Kainz, J. Chen, and M. Levoy. Burst photography for high dynamic range and low-light imaging on mobile cameras. ACM Transactions on Graphics, 35(6), 2016.


## <a id="head8"></a>Appendix and FAQ

:::info
**Find this document incomplete?** Leave a comment!
:::

<!-- ###### tags: `Documentation` -->



[1]: https://web.stanford.edu/~wandell/data/papers/2017-L3-IEEE-Jiang.pdf
[2]: http://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16-preprint.pdf
[3]: https://neptune.ai/blog/moving-from-tensorflow-to-pytorch 
[4]: https://arxiv.org/abs/1502.03167