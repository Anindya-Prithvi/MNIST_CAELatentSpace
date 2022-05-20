# Objective
Observe the Latent space of Convolutional AutoEncoder for a classification dataset (MNIST)

# Architecture of Auto Encoder
The encoder consists of multiple convolutions and batchnormalizations only. We scale a 28x28 to 7x1 vector.  
The decoder is again transposed convolutions and batchnormalizations, i.e. 7x1 to 28x28.  
Optimizers/LR/decays used have not been hyperparameter optimized yet. Therefore the current loss is 0.02 for 50 epochs.  
![CAE Output](https://github.com/Anindya-Prithvi/MNIST_CNNLatentSpace/blob/main/assets/CAEresult.png)

# To Observable
We encode the training images to our latent space as usual. However we then apply LDA transformations with `n_components=2`. Plotting the transformed data we could observe well formed clusters (refer notebook). Also, there were obviously some overlapping ones, to resolve them, a 3D LDA transformation was obtained. The clusters were amazingly distinguishable with almost 0 overlap.  
## 2D LDA result.  
![2D LDA result](https://github.com/Anindya-Prithvi/MNIST_CNNLatentSpace/blob/main/assets/LDA2.png)
## 3D LDA plot © [Aflah](https://github.com/aflah02/)  
[3D LDA result - Interactive (Click)](https://anindya-prithvi.github.io/filehost/plotlypage.html)  
![image](https://user-images.githubusercontent.com/29653551/169578739-40f9c343-d80b-4479-855b-9a4d4062adc5.png)


# Observations
1. We can generate multiple fake samples given the latent space encoding distributed on multivariate gaussian distribution with parameters from MLE estimates of the training data (of a specific class) in latent space. 
2. Latent space was initally made keeping in mind a 7-segment decoder. But guess what (HINT: Look at the means)
3. All of the following is synthetic data  
   ![synthesis](https://github.com/Anindya-Prithvi/MNIST_CNNLatentSpace/blob/main/assets/synthetic.png)  
   ```py
   array([
       3, 2, 2, 2, 7, 6, 6, 2, 2, 2, 2, 1, 2, 6, 2, 2, 
       2, 2, 6, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 1, 2, 
       8, 2, 2, 2, 2, 6, 2, 2, 6, 0, 7, 2, 0, 2, 2, 9, 
       5, 2, 2, 2, 5, 2, 2, 2, 6, 6, 2, 2, 2, 2, 2, 2, 
       2, 2, 2, 2, 4, 8, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 
       6, 2, 8, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 8, 3, 2, 
       2, 2, 6, 2, 8, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 
       2, 6, 2, 2, 2, 2, 2, 2, 2, 8, 2, 6, 8, 2, 2, 2], dtype=int64)
