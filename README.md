## 📌 Introduction

A few months ago, I completed the **Machine Learning Specialization** course. Now, I’m revisiting and revising those machine learning concepts, while also diving deeper into **deep learning**. I’ll be sharing my daily progress here as I go along.

---

## 📚 Resources & Progress

| Title                                                                                      | Progress       |
|--------------------------------------------------------------------------------------------|----------------|
| [Machine Learning – Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning) | ✅ Completed   |
| [Pytorch Tutorial - Daniel Bourke (Youtube)](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=14939s) | ⏳     |
---

### Day 1

Today I watched a back propagation video by 3Blue1Brown and practiced some calculus behind it.

Also, as I am learning pytorch, I read some basic documentations and I made a toy dataset to try training a linear regression model. I have learned that every class in a neural network has to subclasss nn.Module in pytorch. As of now, I wrote a basic code to create a Linear Regression model class with the parameters ( just forward propagation). Tommorow I will be diving deeper into it. 

<img width="555" height="480" alt="image" src="https://github.com/user-attachments/assets/9f3f1eaf-e316-461c-8d36-b17cebf8c522" />   <img width="661" height="297" alt="image" src="https://github.com/user-attachments/assets/099eac33-3b05-4542-bafa-e91ccb3db057" />


### Day 2 

I continued another video from 3blue1Brown and it was about transformer, the tech behind the transformer. I learned that in LLMs, the innitial prompt is taken as a token and the next word is predicted and this continues. The first step would be embedding, assigning the each words/toekns or any symnbol as a vector in a high dimentional vectorspace, 

<img width="1176" height="626" alt="image" src="https://github.com/user-attachments/assets/938ca7dc-f3f9-4321-9903-4c6f8d795416" />

The next word is chosen by using the softmax activation ( the vector with the highest probablity is chosen ). Also, we can determine the temperature by which the model choses more creative answerers but it decreases the accuracy of the prediction.

<img width="935" height="608" alt="image" src="https://github.com/user-attachments/assets/4da7d9a3-4eee-40d2-b240-86bf5685dcfc" />


### Day 3

Today I continued the Pytorch course by Daniel Bourke on Youtube. Previously I wrote code to generate the toy dataset and split into test, train dataset and next I made a Class subclassing nn.mocule to calculate the Linearregression model.

I wrote code to predict the output and calculated the loss function to change the parameter for better output.


### Day 4

Today I continued the pytorch video and I made a full training loop involving loss forward pass, loss calculation, optimization performing gradient descent. 

<img width="749" height="578" alt="image" src="https://github.com/user-attachments/assets/8c4e5c7f-4762-4244-a4aa-2ed070e8b7d7" />

 It was a toy dataset with random parameters so the predictions at first were terrible. After training the model the predictions were better.

 <img width="939" height="652" alt="image" src="https://github.com/user-attachments/assets/bd0d345b-e908-4def-b680-5a21cdb10664" />  <img width="938" height="654" alt="image" src="https://github.com/user-attachments/assets/1daa7394-a71c-4189-9283-ceb4bd060c7c" />


