This is more like a report.  
# Objective
Observe the Latent space of Convolutional AutoEncoder for a classification dataset (MNIST)

# Architecture of Auto Encoder
The encoder consists of multiple convolutions and batchnormalizations only. We scale a 28x28 to 7x1 vector.  
The decoder is again transposed convolutions and batchnormalizations, i.e. 7x1 to 28x28.  
Optimizers/LR/decays used have not been hyperparameter optimized yet. Therefore the current loss is 0.02 for 50 epochs.

# To Observable
We encode the training images to our latent space as usual. However we then apply LDA transformations with `n_components=2`. Plotting the transformed data we could observe well formed clusters (refer notebook). Also, there were obviously some overlapping ones, to resolve them, a 3D LDA transformation was obtained. The clusters were amazingly distinguishable with almost 0 overlap.

# Observations
1. We can generate multiple fake samples given the latent space encoding distributed on multivariate gaussian distribution with parameters from MLE estimates of the training data (of a specific class) in latent space. 
2. Latent space was initally made keeping in mind a 7-segment decoder. But guess what (HINT: Look at the means)
