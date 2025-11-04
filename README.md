## 📌 Introduction

A few months ago, I completed the **Machine Learning Specialization** course. Now, I’m revisiting and revising those machine learning concepts, while also diving deeper into **deep learning**. I’ll be sharing my daily progress here as I go along.

---

## 📚 Resources & Progress

| Title                                                                                      | Progress       |
|--------------------------------------------------------------------------------------------|----------------|
| [Machine Learning – Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning) | ✅ Completed   |
| [Pytorch Tutorial - Daniel Bourke (Youtube)](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=14939s) | ⏳     |
---

# Day 1

## Machine Learning Revision

### Power Transformer

I learned about power transformer class in Scikit-learn. I basically got to know that in order to train machine learning models our data must have normal distribution so in order to make our data normal, one of the most efficient way it power transformation. I used it practically on a concrete dataset from kaggle. Most of the data were not normally distributed, one of those can be seen below through distplot and QQ-Plot below.

<img width="1150" height="393" alt="image" src="https://github.com/user-attachments/assets/bb4a407e-4b59-4ff6-b41b-df220a05bf64" /> 

I used linear regression model without any transformation first and got the accuracy of ~62% and after using box-coz transformation I got 80% accuracy, however on crossvalidating it was around 46%.
Later on using 'Yeo-Johnson' my results were better than previous I got 81% accuracy and ~60% on cross-validationg. 

One example before and after the Yeo-Johnson transformation: 

<img width="1662" height="545" alt="image" src="https://github.com/user-attachments/assets/885e85be-d880-4229-ae7e-28b1309891ef" />

Next, I learned about Binarizer class in scikit-learn. Sometimes we might neet to map a value greater than a threshold to 1 and lower than that would be zero, in that case we can use this class in order to transform the features. 

### Convolution Neural Network

Object Localization

I learned how objection localization is done. We need bounding boxes which will help the model understand where exactly is the object located in the image.

<img width="1893" height="1079" alt="image" src="https://github.com/user-attachments/assets/a970bccf-1dd7-42cf-8239-28cb477824d5" />

We need output like y = [
                        Pc (probablity)
                        bx (x coordinate of the center of the bounding box)
                        by (y coordinate of the center of the bounding box)
                        bh (height of the bounding box)
                        bw (width of the bounding box)
                        c1 - label 1
                        c2 - label 2
                        c3 - label 3
                        ]



# Day 2

## Machine Learning Revision

### Handeling Mixed Variables

I started with handeling mixed variables today, where I took a toy data with mainly 2 kinds of problem with mixed variables. Firstly, one with the numeric and caterogical values in same column. For that we can use to_numeric() function in pandas to extract numeric data and make a new column, and for the remaining categorical data we can fillup with the categorical data from the orignal data ( for columns which has NaN values in the new numeric column). Here's an example: 

<img width="1070" height="615" alt="image" src="https://github.com/user-attachments/assets/e1bb9410-4be7-4085-b9ba-a9edf383fd26" />  <img width="1082" height="383" alt="image" src="https://github.com/user-attachments/assets/d2f14baf-2a90-49ba-b20a-b5c627eebff6" />

Next, for the columns where numeric and categorical data we have to seperate both numeric and categorical and put them in a seperate columns. 

<img width="1399" height="645" alt="image" src="https://github.com/user-attachments/assets/fe356990-e980-4109-b885-c62920271dee" />

Also learned about how to handel date and time, I learned to use a pandas function pd.to_datetime() which allows us to change the object datatype to datetime datatype. It helps us to perform other important operations and extract datas. 

<img width="558" height="596" alt="image" src="https://github.com/user-attachments/assets/6a636ff3-ada9-40c7-85d1-c29a703bc05c" />     <img width="985" height="621" alt="image" src="https://github.com/user-attachments/assets/b205781f-5d8c-4fe7-a400-813c0fab1f1f" />

## Convolution Neural Network

So today I went deep into understanding how object localization is done, first landmark detection where we use CNN to locate where a certain object is present in the image. And next, about bounding boxes and how we spicify where exactly is the object located. For that, first train model with bounding boxex and with the stride and label it as 1 or 0, and bounding box is created whereever it is true, first the center of the bounding box bx,by and the width and hight of the bounding box. 

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/661af6b9-f130-403e-a589-7532f9e462b0" />


We use convolution instead of FC because it gives the prediction for all the regions at once. 



# Day 3

## Machine Learning Revision

### Handeling Missing data

I learned how we can deal with the missing data, there are several ways to deal with them such as Removing (CCA - Complete Case Analysis), and Imputation. There are other sub-parts of imputation like univariate and multivariate imputation which I will be studying tommorow. 

Today however I learned about a least important topic still useful sometimes, in this method we complete remove the whole row if one value in any of the column is missing and we call it CCA. In order to use this method we need to make sure that the values that are missing are 'completely at random' we cannot do it if top 50 or bottom 50 values are missing. And the other condition is that we generally use this only if upto 5% data is missing (at max). 

<img width="774" height="695" alt="image" src="https://github.com/user-attachments/assets/89db1e4b-c8ad-4bef-8826-4322c76c1bf6" />

Here I used a dataset and performed CCA on one of the column and the red is the orignal data ( slightly on the top ) and blue is after performing CCA. This shows that the data were missing completely at random.


# Day 4

Today I learned about Insertion over Union (IoU) in object detection. So here I got to know that we have the actual bounding box and the predicted bounding box and the total area of both of the boxes including where they overlap is the Union. And the area where they intersect is intersection. When we divide the area of the intersection with the total area of the bounding box, we get certain number and if greater than 0.5, we consider that true prediction and false otherwise. 

<img width="1899" height="1012" alt="image" src="https://github.com/user-attachments/assets/13a23123-71c4-4005-9054-7a29f8d4af62" />


# Day 5

## Non-Max Supression

Non-Maximum Suppression (NMS) is a post-processing step in object detection that eliminates redundant bounding boxes for the same object. Object detectors often predict multiple boxes with high overlap—NMS helps by keeping only the most confident one.

<img width="1731" height="968" alt="image" src="https://github.com/user-attachments/assets/b4cfda9b-39b5-43a2-a125-24822b5102b0" />

Steps:

Sort boxes by confidence score.

Select the highest score box.

Remove boxes with high IoU (e.g., > 0.5) with the selected box.

Repeat until no boxes remain.



# Day 6

Anchor boxes are predefined bounding boxes with different sizes and aspect ratios used in object detection models like YOLO, SSD, and Faster R-CNN.

They help detect multiple objects of different shapes at the same location.


<img width="1746" height="974" alt="image" src="https://github.com/user-attachments/assets/e9626686-ea0c-423c-bc4c-af3c58462e45" />


### Key Points:
- Each grid cell predicts offsets for several anchor boxes.
- Anchor boxes have fixed sizes (e.g., 1:1, 2:1, 1:2).
- The model learns to adjust these boxes to match real objects.
- Useful for handling overlapping and varied-shaped objects.

**Example:**  
If a grid cell has 3 anchor boxes, it can predict 3 different object candidates at once.



# Day 7


## YOLO (You Only Look Once)


<img width="1102" height="625" alt="image" src="https://github.com/user-attachments/assets/d636da91-4a7d-403b-860b-719032626df9" />


YOLO (You Only Look Once) is a real-time object detection algorithm that frames detection as a single regression problem, directly predicting bounding boxes and class probabilities from an input image in one evaluation. It divides the image into a grid, and each grid cell predicts a fixed number of bounding boxes along with confidence scores and class probabilities. Unlike traditional methods that use region proposals followed by classification, YOLO performs both tasks simultaneously, making it extremely fast and suitable for real-time applications. It learns global features of the image, resulting in fewer false positives. However, YOLO can struggle with detecting small or overlapping objects, especially when they fall within the same grid cell. Despite this, its balance of speed and accuracy has made it one of the most widely used object detection algorithms in practice.



# Day 8

## Semantic Segmentation with U-Net

U-Net is a convolutional neural network architecture designed for semantic segmentation, where the goal is to classify each pixel of an image into a category. It follows a U-shaped architecture consisting of a contracting path (encoder) that captures context and a symmetric expanding path (decoder) that enables precise localization. Skip connections between encoder and decoder layers help recover spatial information lost during downsampling. Originally developed for biomedical image segmentation, U-Net performs well even with limited data and has become a popular choice for tasks like medical imaging, satellite imagery analysis, and road segmentation. Each output pixel is assigned a class label, resulting in a full-resolution segmentation map.

<img width="1893" height="1001" alt="image" src="https://github.com/user-attachments/assets/6b573cfc-8cf8-4189-bbda-87b3b40a0b4c" />  <img width="1230" height="620" alt="image" src="https://github.com/user-attachments/assets/4238b670-380c-4d6f-b10a-8dd1712b6cb4" />



# Day 9

## Transpose Convolution in U-Net

<img width="1894" height="996" alt="image" src="https://github.com/user-attachments/assets/c44d2311-c902-4ad6-bc9e-d894c1610636" />    <img width="1916" height="1074" alt="image" src="https://github.com/user-attachments/assets/5dd8c826-b39d-4064-8f93-147a78550d66" />



In U-Net, transpose convolutions (also known as deconvolutions or up-convolutions) are used in the expanding path (decoder) to upsample feature maps and recover spatial resolution lost during downsampling. Unlike simple interpolation methods, transpose convolutions are learnable layers that can learn how to upsample in a task-specific way. This allows the network to reconstruct high-resolution output with better precision. Each transpose convolution is typically followed by concatenation with corresponding encoder features via skip connections, enabling the model to combine semantic and spatial information for accurate pixel-wise predictions in semantic segmentation tasks.



# Day 10 


Face recognition aims to identify or verify individuals from images by comparing facial features. Traditional models require large datasets, but one-shot learning enables recognition using only one example per class. This is achieved using a Siamese Network, which consists of two identical neural networks sharing weights, trained to learn a similarity function rather than classifying directly. Given a pair of images, the network learns to output whether they represent the same person by comparing their feature embeddings. This approach is highly effective in applications like face verification, where collecting many samples per person is impractical.

<img width="1322" height="715" alt="image" src="https://github.com/user-attachments/assets/b84c36ee-26f3-44ce-a067-ee4eab871f95" />

<img width="1257" height="675" alt="image" src="https://github.com/user-attachments/assets/74a813ab-12e6-489c-875f-7ee3043a2c42" />



# Day 11

## Triplet Loss

Triplet Loss is a loss function commonly used in face recognition and metric learning tasks. It encourages a neural network to learn an embedding space where images of the same identity are closer together and images of different identities are farther apart. Each training sample is a triplet consisting of: an anchor image, a positive image (same identity), and a negative image (different identity). The loss minimizes the distance between the anchor and positive, while maximizing the distance between the anchor and negative by at least a predefined margin. This helps the model learn discriminative and compact feature representations for tasks like one-shot learning and verification.

<img width="1917" height="1079" alt="image" src="https://github.com/user-attachments/assets/133a39e7-573f-4d22-909b-682705ebdaa8" />



# Day 12

## Neural Style transfer

Neural Style Transfer (NST) is a deep learning technique that blends the content of one image with the style of another to create a visually artistic output. It uses a pre-trained convolutional neural network (typically VGG19) to extract content features from a source image and style features (like colors, textures, and patterns) from a style image. The algorithm then optimizes a third image to minimize both content loss (difference from content image) and style loss (difference from style image). NST is widely used in AI art generation, allowing neural networks to recreate images in the style of famous painters like Van Gogh or Picasso.

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/37be82c6-0828-4644-8d01-39db913001ec" />



# Day 13

## What are deep ConvNets learning?

Today I learned something really cool about how deep convolutional neural networks (ConvNets) actually work. Instead of just memorizing images, they learn in layers—starting with really simple things like edges and textures, then gradually moving on to more complex patterns like shapes and object parts. By the time you get to the deeper layers, the network can recognize entire objects like faces or animals. What surprised me is that the network figures out these features on its own during training—no one tells it what to look for. It also makes sense now why transfer learning works so well: the early layers learn general visual patterns that are useful for lots of different tasks, while only the later layers need to be retrained. It’s pretty amazing how these networks can build such a detailed understanding of images just from raw pixel data.

<img width="1912" height="1079" alt="image" src="https://github.com/user-attachments/assets/d5eca5f8-c676-4e6e-b683-f23d8c9d8124" />



# Day 14

## Style cost function

Today I learned about style cost function. The style cost function in Neural Style Transfer measures how well the generated image captures the style of a given style image. It does this by comparing the correlations between feature maps in both images, which are represented using something called a Gram matrix. These feature maps are taken from multiple layers of a pretrained convolutional neural network (like VGG). The Gram matrix captures the textures, colors, and patterns present in an image by looking at how different filters activate together. To compute the style cost, the algorithm calculates the Gram matrices of both the style image and the generated image at several chosen layers, then measures the squared difference between them. This difference tells us how similar the styles are. The overall style cost is obtained by summing the losses from each layer, often with weights to control their influence. Minimizing this style cost during training helps the generated image adopt the style characteristics—like brush strokes or color patterns—of the style image.

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/42cc790d-0e43-4f75-898b-1dafb6ba552d" />



# Day 15

## 1D and 3D Generalizations

1D and 3D convolutions are generalizations of the standard 2D convolution used in image processing, adapted to work with different types of data. A 1D convolution is typically used for sequential data like time series, audio signals, or text. It operates along a single spatial or temporal dimension and is useful for detecting local patterns in sequences, such as repeated structures or temporal dependencies. On the other hand, a 3D convolution is used when the data has three spatial dimensions, such as volumetric data in medical imaging (like MRI or CT scans) or video data where time is treated as the third dimension alongside height and width. In 3D convolution, the filter slides over height, width, and depth, allowing the model to capture spatial and temporal patterns simultaneously. Both of these generalizations help convolutional networks adapt to different types of structured data beyond just 2D images.

<img width="1331" height="745" alt="image" src="https://github.com/user-attachments/assets/bc7f368b-f202-4f4f-ab19-d93e3ae4efd2" />  <img width="1343" height="669" alt="image" src="https://github.com/user-attachments/assets/9d1ec5ce-517e-4709-aee2-48a34b7d7b98" />



# Day 16

# Cost function

In the last week of the CNN course in the Deep Learning Specialization, I reviewed how Neural Style Transfer works by combining the content of one image with the style of another. A key part of this is the content cost function, which helps the generated image preserve the structure of the original content image. Instead of comparing pixel values, it compares feature activations from a pretrained CNN (like VGG-19), ensuring the high-level content remains similar. This content cost is combined with style cost to guide the generation process using a total cost function.

<img width="1732" height="949" alt="image" src="https://github.com/user-attachments/assets/59bde3a1-733a-46aa-abc3-0c4b5021b776" />    <img width="1909" height="994" alt="image" src="https://github.com/user-attachments/assets/3063963a-9003-4bd0-b980-6daf18196460" />



# Day 17

## Face Verification and Binary Classification

Today I learned the difference between face verification and binary classification and how they are connected in deep learning tasks. Face verification is the process of determining whether two facial images belong to the same person. It doesn't try to recognize who the person is (like face recognition does), but simply checks if the identities match. Interestingly, this is a type of binary classification problem because the model’s output is just one of two options: either the faces match (1) or they don't (0). The model learns to compare two images and predict whether they are of the same individual by analyzing facial features and their similarities. This task often uses architectures like Siamese networks, which are designed to compare pairs of inputs and learn a similarity score. So in essence, face verification applies binary classification in a very specific and meaningful contex

<img width="1919" height="1078" alt="image" src="https://github.com/user-attachments/assets/77a8494d-5efa-4a23-966f-e2d3d9a031eb" />
