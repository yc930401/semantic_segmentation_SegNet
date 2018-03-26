# semantic_segmentation_SegNet

A keras implementation of SegNet for semantic segmentation tasks.

## Introduction

The idea to build a SegNet was arisen when I learnt Stanford CS231. I thought it shouldn't be a difficult task, because the idea was very clear to me.
The only difference between a CNN classification network and a SegNet is to use Upsampling to recover the original image dimensionality. And it should be
able to deal with images with various sizes. The network structure of a SegNet is shown in the figure below. </br>
![SegNet](/segnet_architecture.png) </br>
However, I encountered some difficulties when building the system.

## Methodology

1. Design the network architecture follow the figure above
2. Find image data with semantic labels
3. Build a generator to preprocess and generate data for training
4. Show result

## Difficulties

1. SegNet should be able to deal with images with different sizes, but all tutorials availabel on the internet tell me to resize the input images to be the same size.
I try to change the input size to (None, None, 3). It seems to tackle this problem.
2. Upsampling2D layer cannot always get the origianl size back. For example, if the original size is (3,3), after one maxpooling layer, the size is (1,1),
 and after applying a Upsampling layer, the output should be (2,2). There is no way for me to recover the size the the input image if the size of image is odd.
 For this problem, I think we can either resize the image, I'm not sure if resizing will result in pool resolution.
3. Training the system may need a large amount of time, becasue there are 26 conv layers and many other layers in the network. The networks I trained previous
 only contains several layers. I didn't train the network, so I cannot provide any result here.

## References:
https://github.com/mrgloom/awesome-semantic-segmentation </br>
https://github.com/ykamikawa/SegNet/blob/master/SegNet.py </br>
https://github.com/preddy5/segnet </br>
https://github.com/imlab-uiip/keras-segnet </br>
https://github.com/0bserver07/Keras-SegNet-Basic </br>
