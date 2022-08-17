In my thesis I want to investigate the effectiveness of different manifold manipulation techniques on the performance of transformer based time series forecasting methods. Transformer models can work with time series , however, those sequence-to-sequence models rely on neural attention between timesteps, which allows for temporal learning but fails to consider distinct spatial relationships between variables inside the timestep. To solve this problem, Jeng et al. (2022) have introduced a new embedding methodology (Spacetimeformer) to capture the spatiotemporal relationship between variables. 
As part of my thesis, I want to compare 4 different models: 

Transformer
Transformer + Manifold Mixup
Spacetimeformer 
Spacetimeformer + Manifold Mixup

Since time-series data is not easily interpretable by humans, I will use PCA and t-SNE to map the multi-dimensional output sequence vectors into two dimensions to visually observe the similarity in the distribution of the synthetic data and real data instances.
