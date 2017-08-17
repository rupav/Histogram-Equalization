# Image-Histogram-Equalizaion

This repository is for implementation of Histogram Equalization for its application in the field of Machine Learning, AI, NeuralNets etc.! 

We often come across images whose contrast is not good enough for visualizing or using in Machine Learning classification tasks. So to enhance brightness/contrast I have used the technique called Global Histogram Equalization.

**# Gloabal Histogram Equalization on Colored Image!**

**## Original Image to work On**
![Original Image](https://github.com/rupav/Image-Histogram-Equalizaion/blob/master/Thanos.jpg)

_**## Histogram of intensities of input Image**_
![input Histogram of Image](https://github.com/rupav/Image-Histogram-Equalizaion/blob/master/input_histogram.png)

_**## Histogram of intensities of Transformed Image**_
![Transformed Image Histogram](https://github.com/rupav/Image-Histogram-Equalizaion/blob/master/output_histogram.png)
### See how our transformation function distributed the pixel counts to different bins(or intensities) evenly!
### Thats what GHE is all about

_**## Projecting transformed histogram into the original image!**_
![GHE transformed Image](https://github.com/rupav/Image-Histogram-Equalizaion/blob/master/transformed_image.png)
### See the difference... Better contrast than original!

**# Conclusion!**

![Result](https://github.com/rupav/Image-Histogram-Equalizaion/blob/master/Final%20Analysis.png)

**## As we can see, GHE on RGB image directly provided better result on this image.**
**## Otherwise YCbCr image should have provided better result on coloured image.**
