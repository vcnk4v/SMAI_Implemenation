# Statistical Methods in Artificial Intelligence

Assignment 1

2022101104

## **K-Nearest Neighbors**

**2.2.1**

The dataset includes several features, but `track_ids`, `artists`, `album_name`, and `track_name` do not contribute to predicting `track_genre` and can be excluded.

To analyze the distribution of a specific feature, including identifying outliers and skewness, we can use violin and box plots. Additionally, individual features have been visualized using histograms.

Histograms:

![histograms.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/histograms.png)

Boxplots:

![boxplots.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/boxplots.png)

Violin plots:

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image.png)

Explicit:

![explicit_distribution.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/explicit_distribution.png)

Correlation:

![correlation_matrix.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/correlation_matrix.png)

According to this heatmap, some features in order of importance could be acousticness, valence, liveness, popularity. 

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%201.png)

Miscellaneous analysis such as energy distribution across genres is also performed. It can be seen that energy levels for some genres lie in very specific ranges with only a few outliers and can be useful for prediction.

**2.3 KNN Implementation**

The `KNN` class implements a k-Nearest Neighbors algorithm with support for multiple distance metrics: Euclidean, Manhattan, and Cosine. The class can operate in two modes: optimized and naive. The optimized mode leverages `numpy`'s efficient vectorized operations for distance computations, significantly improving performance. In contrast, the naive implementation uses explicit for-loops, which are less efficient. The `predict` method measures inference time, providing insights into the performance of the model.

**Dataset**: spotify.csv

**Preprocessing**: In the preprocessing step, numerical features such as `popularity`, `duration_ms`, `danceability`, and others are normalized to ensure they are on a similar scale, which can improve model performance. Categorical features like `track_genre` are encoded using a factorization approach, converting them into numerical values suitable for machine learning algorithms. Irrelevant features, including `track_id`, `artists`, `album_name`, `track_name`, and `Unnamed: 0`, are dropped as they do not contribute to predicting the `track_genre`. The dataset is then shuffled to ensure randomness before being saved for future use. Finally, features and the target variable are separated for model training.

**2.4 Hyperparameter Tuning**

**2.4.1**

1. **Best {k, distance metric} :** 

K: 24, Metric: manhattan
Accuracy: 0.255
Precision: 0.24589041508480547
Recall: 0.25498415989593615
F1 Score Macro: 0.25035473569115724
F1 Score Micro: 0.2119815668202765

1. **Ordered rank list:**

| **k** | **Distance Metric** | **Accuracy** |
| --- | --- | --- |
| 24 | manhattan | 0.255000 |
| 18 | manhattan | 0.252018 |
| 22 | manhattan | 0.252001 |
| 23 | manhattan | 0.251316 |
| 19 | manhattan | 0.251140 |
| 20 | manhattan | 0.251002 |
| 21 | manhattan | 0.242614 |
| 17 | manhattan | 0.242351 |
| 16 | manhattan | 0.241737 |
| 15 | manhattan | 0.241649 |
1. **k vs accuracy, Distance metric: Manhattan**

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%202.png)

1. **Dropped all columns one-by-one each and checked for improvement in accuracy**

| **Column Dropped** | **Accuracy** |
| --- | --- |
| acousticness | 0.2482 |
| valence | 0.2515 |
| liveness | 0.2650 |
| popularity | 0.1921 |
| duration_ms | 0.2371 |
| explicit | 0.2529 |
| danceability | 0.2442 |
| energy | 0.2547 |
| speechiness | 0.2529 |
| key | 0.2645 |
| loudness | 0.2528 |
| instrumentalness | 0.2489 |
| tempo | 0.2420 |

Based on the results from dropping individual columns and observing the impact on accuracy, it is evident that certain features significantly influence the model's performance in predicting `track_genre`. When the `popularity` column was dropped, the accuracy dropped notably to **0.1921**, indicating its critical importance in the model. In contrast, removing features like `valence`, `explicit`, and `energy` showed minor impacts, with accuracies around **0.25**, suggesting that these features still contribute meaningfully to the model.

The highest accuracy was observed when dropping the `key` column, achieving **0.2645** accuracy, closely followed by dropping the `liveness` feature, resulting in **0.265** accuracy. This implies that while `key` and `liveness` provide some predictive power, they are not as crucial as other features in the dataset.

From these observations, it can be concluded that the most effective feature combination for predicting `track_genre` should include `popularity`, `duration_ms`, `danceability`, and `instrumentalness`, among others. Features such as `key` and `liveness` might be less essential and could potentially be deprioritized or considered for exclusion in specific scenarios to optimize the model's complexity without significantly affecting predictive performance.

### 2. Testing Combinations

Accuracy taking only one column:

| Feature | Accuracy |
| --- | --- |
| `['popularity']` | 0.0501 |
| `['duration_ms']` | 0.0518 |
| `['explicit']` | 0.0144 |
| `['danceability']` | 0.0317 |
| `['energy']` | 0.0396 |
| `['key']` | 0.0108 |
| `['loudness']` | 0.0472 |
| `['mode']` | 0.0079 |
| `['speechiness']` | 0.0343 |
| `['acousticness']` | 0.0488 |
| `['instrumentalness']` | 0.0328 |
| `['liveness']` | 0.0259 |
| `['valence']` | 0.0292 |
| `['tempo']` | 0.0633 |
| `['time_signature']` | 0.0104 |

Popularity, duration and temp seem to be the most important features. Liveliness, valence and key the least.

**2.5 Optimization**

**2.5.1** 

### Vectorization vs. Naive Approach

- **Vectorization**: Achieved through `numpy`'s parallelized operations, significantly improving performance.
- **Naive Model**: Utilized simple for loops, which were less efficient compared to vectorized operations.

### Inference Time Comparison

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%203.png)

### Inference Time Comparison

- **Best KNN Model**:
    - **Parameters**: k=24, Distance Metric: Manhattan
    - **Inference Time**: 0.0088 seconds
- **Optimized KNN Model**:
    - **Parameters**: k=3, Distance Metric: Manhattan
    - **Inference Time**: 0.0087 seconds
- **Naive KNN Model**:
    - **Parameters**: k=3, Distance Metric: Manhattan
    - **Inference Time**: 0.2482 seconds
- **Default sklearn KNN Model**:
    - **Parameters**: k=3, Distance Metric: Manhattan
    - **Inference Time**: 0.0005 seconds

**Key Observations**

- The best and optimized model with k=24 and Manhattan distance achieved the lowest inference time.

**Inference time vs train dataset size** 

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%204.png)

Inference time: Time taken to inference entire dataset

Observation: sklearn also uses similar (and better) vectorisation and parallelisation techniques to reduce inference time. As the train size increases, initialKNN starts performing worse in comparision to other knn models. Thus clearly performance of initialKNN is a function of the training set size. Other models’ performances are fairly independent of the size of training dataset.

### EDA on Spotify-2

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%205.png)

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%206.png)

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%207.png)

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%208.png)

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%209.png)

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%2010.png)

The feature distributions in the Spotify-2 dataset are quite similar to those in the original Spotify dataset. Based on the distribution patterns observed, one would expect the model to perform comparably well on Spotify-2 as it did on the original dataset.

However, the model's performance on the Spotify-2 dataset is significantly worse than anticipated. Specifically, the accuracy drops to approximately 25% when data is concatenated and normalized together. In contrast, separate Gaussian normalization for each dataset results in better performance. This suggests that while the distributions are similar, the method of normalization plays a crucial role in model performance.

Concatenating datasets and then applying a single normalization process ensures that the scaling parameters (mean and standard deviation) are consistent across all data. This avoids the risk of discrepancies that can arise from normalizing each dataset separately with different parameters.

When normalization is applied to the concatenated data, the model can learn features that are more representative of the entire data distribution. This can improve the model's ability to generalize from training to test data, especially when the datasets are related but not identical.

K: 24, Metric: manhattan
Accuracy: 0.01
Precision: 0.009610044913155836
Recall: 0.01049959679331456
F1 Score Macro: 0.01003514614796088
F1 Score Micro: 0
Accuracy:  0.01

**After combining (concatenating train, val, test) and normalising:**

Accuracy: 0.24921052631578947
Precision: 0.24187338707433897
Recall: 0.25007087931859284
F1 Score Macro: 0.2459038339157493
F1 Score Micro: 0.07531380753138076
Accuracy:  0.24921052631578947

## Linear Regression

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%2011.png)

**3.1.1**

Degree = 1

| **Train MSE** | **Test MSE** | **Train Variance** | **Test Variance** | **Train Std Deviation** | **Test Std Deviation** |
| --- | --- | --- | --- | --- | --- |
| 0.3737 | 0.2517 | 0.3736 | 0.2503 | 0.6113 | 0.5003 |

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%2012.png)

**3.1.2**

Degree > 1

| **Degree** | **Train MSE** | **Test MSE** | **Train Variance** | **Test Variance** | **Train Std Deviation** | **Test Std Deviation** |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.2532 | 0.1002 | 0.2532 | 0.0999 | 0.5032 | 0.3161 |
| 3 | 0.0944 | 0.0683 | 0.0944 | 0.0654 | 0.3072 | 0.2558 |
| 4 | 0.0926 | 0.0592 | 0.0926 | 0.0566 | 0.3042 | 0.2380 |
| 5 | 0.0813 | 0.0645 | 0.0812 | 0.0613 | 0.2850 | 0.2476 |

B**est Degree: 4** with Test MSE: 0.05918484981382568

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%2013.png)

**Experimentation:**

**Different Learning Rates**

| Set | Learning Rate | MSE | Standard Deviation | Variance |
| --- | --- | --- | --- | --- |
| Train | 1e-06 | 1.8486 | 1.3597 | 1.2501 |
| Test | 1e-06 | 1.3295 | 1.1530 | 0.8471 |
| Val | 1e-06 | 1.2459 | 1.1162 | 0.8349 |
| Train | 0.0001 | 1.5641 | 1.2506 | 1.1528 |
| Test | 0.0001 | 1.0837 | 1.0410 | 0.7602 |
| Val | 0.0001 | 1.0201 | 1.0100 | 0.7625 |
| Train | 0.01 | 0.3736 | 0.6112 | 0.3736 |
| Test | 0.01 | 0.2547 | 0.5047 | 0.2533 |
| Val | 0.01 | 0.2703 | 0.5199 | 0.2466 |
| Train | 0.1 | 0.3737 | 0.6112 | 0.3736 |
| Test | 0.1 | 0.2517 | 0.5003 | 0.2503 |
| Val | 0.1 | 0.2705 | 0.5201 | 0.2468 |
| Train | 1 | 2.5153e+22 | 158597940624.9787 | 1.6236e+20 |
| Test | 1 | 2.5679e+22 | 160247789432.3625 | 1.8250e+20 |
| Val | 1 | 2.5032e+22 | 158216031843.3959 | 1.3334e+20 |

Best learning rate: 0.01

On performing a similar experiment for degrees > 1, best learning rate was found to be 0.01.

 

**Different Seeds**

| **Seed** | **Converged After Iterations** | **Train MSE** | **Test MSE** | **Train Variance** | **Test Variance** | **Train Std Deviation** | **Test Std Deviation** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 1000 | **0.0926** | **0.0592** | 0.0926 | 0.0566 | 0.3042 | 0.2380 |
| 0 | 1000 | 0.0988 | 0.0642 | 0.0988 | 0.0608 | 0.3143 | 0.2465 |
| 123 | 1000 | 0.2247 | 0.1010 | 0.2247 | 0.1010 | 0.4740 | 0.3179 |
| 999 | 1000 | 0.2097 | 0.0931 | 0.2096 | 0.0931 | 0.4579 | 0.3051 |
| 2024 | 1000 | 0.1865 | 0.0848 | 0.1864 | 0.0845 | 0.4318 | 0.2907 |

Best seed : 42

Though it takes same number of epochs to converge, the mse is less for seed as 42. If we start with an opposite or skewed curve from the correct, it will take longer for the model to reduce MSE. Hence, inititalisation makes a difference.

**3.2 Regularisation**

**3.2.1**

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%2014.png)

**Without Regularisation:**

| **Degree** | **Train MSE** | **Test MSE** | **Train Variance** | **Test Variance** | **Train Std Dev** | **Test Std Dev** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2230 | 0.1474 | 0.2230 | 0.1474 | 0.4722 | 0.3840 |
| 2 | 0.0227 | 0.0121 | 0.0227 | 0.0117 | 0.1505 | 0.1082 |
| 3 | 0.0378 | 0.0221 | 0.0377 | 0.0220 | 0.1943 | 0.1484 |
| 4 | 0.0250 | 0.0184 | 0.0250 | 0.0184 | 0.1582 | 0.1355 |
| 5 | 0.0129 | 0.0100 | 0.0129 | 0.0100 | 0.1138 | 0.0999 |
| 6 | 0.0197 | 0.0175 | 0.0197 | 0.0167 | 0.1404 | 0.1292 |
| 7 | 0.0181 | 0.0154 | 0.0181 | 0.0148 | 0.1345 | 0.1216 |
| 8 | 0.0154 | 0.0128 | 0.0154 | 0.0124 | 0.1240 | 0.1115 |
| 9 | 0.0153 | 0.0124 | 0.0153 | 0.0121 | 0.1235 | 0.1100 |
| 10 | 0.0144 | 0.0116 | 0.0144 | 0.0114 | 0.1201 | 0.1069 |
| 11 | 0.0157 | 0.0137 | 0.0157 | 0.0133 | 0.1254 | 0.1154 |
| 12 | 0.0176 | 0.0153 | 0.0176 | 0.0147 | 0.1325 | 0.1213 |
| 13 | 0.0292 | 0.0260 | 0.0292 | 0.0246 | 0.1710 | 0.1568 |
| 14 | 0.0144 | 0.0139 | 0.0144 | 0.0138 | 0.1201 | 0.1176 |
| 15 | 0.0158 | 0.0147 | 0.0158 | 0.0146 | 0.1257 | 0.1207 |
| 16 | 0.0145 | 0.0143 | 0.0145 | 0.0143 | 0.1203 | 0.1196 |
| 17 | 0.0145 | 0.0146 | 0.0145 | 0.0146 | 0.1205 | 0.1208 |
| 18 | 0.0164 | 0.0170 | 0.0164 | 0.0169 | 0.1281 | 0.1302 |
| 19 | 0.0156 | 0.0132 | 0.0155 | 0.0131 | 0.1247 | 0.1147 |
| 20 | 0.0173 | 0.0124 | 0.0173 | 0.0124 | 0.1316 | 0.1115 |

**Best Degree: 5** with Test MSE: 0.009985391818613739

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%2015.png)

**With regularisation:**

**L1:**

| **Degree** | **Train MSE** | **Test MSE** | **Train Variance** | **Test Variance** | **Train Std Dev** | **Test Std Dev** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2315 | 0.1543 | 0.2290 | 0.1526 | 0.4785 | 0.3906 |
| 2 | 0.0432 | 0.0197 | 0.0411 | 0.0189 | 0.2029 | 0.1374 |
| 3 | 0.0358 | 0.0158 | 0.0335 | 0.0149 | 0.1831 | 0.1219 |
| 4 | 0.0280 | 0.0131 | 0.0255 | 0.0118 | 0.1597 | 0.1087 |
| 5 | 0.0278 | 0.0147 | 0.0255 | 0.0135 | 0.1597 | 0.1163 |
| 6 | 0.0270 | 0.0273 | 0.0248 | 0.0227 | 0.1575 | 0.1508 |
| 7 | 0.0221 | 0.0214 | 0.0197 | 0.0166 | 0.1404 | 0.1287 |
| 8 | 0.0205 | 0.0186 | 0.0194 | 0.0159 | 0.1394 | 0.1262 |
| 9 | 0.0196 | 0.0168 | 0.0190 | 0.0149 | 0.1379 | 0.1222 |
| 10 | 0.0193 | 0.0158 | 0.0191 | 0.0148 | 0.1383 | 0.1217 |
| 11 | 0.0255 | 0.0239 | 0.0253 | 0.0221 | 0.1590 | 0.1487 |
| 12 | 0.0262 | 0.0264 | 0.0230 | 0.0195 | 0.1515 | 0.1396 |
| 13 | 0.0633 | 0.0655 | 0.0608 | 0.0537 | 0.2466 | 0.2317 |
| 14 | 0.0279 | 0.0307 | 0.0254 | 0.0258 | 0.1593 | 0.1607 |
| 15 | 0.0254 | 0.0271 | 0.0229 | 0.0226 | 0.1514 | 0.1502 |
| 16 | 0.0263 | 0.0303 | 0.0258 | 0.0292 | 0.1607 | 0.1709 |
| 17 | 0.0241 | 0.0251 | 0.0211 | 0.0214 | 0.1452 | 0.1463 |
| 18 | 0.0221 | 0.0250 | 0.0220 | 0.0250 | 0.1485 | 0.1580 |
| 19 | 0.0258 | 0.0232 | 0.0229 | 0.0197 | 0.1513 | 0.1402 |
| 20 | 0.0237 | 0.0176 | 0.0208 | 0.0146 | 0.1442 | 0.1209 |

**Best Degree: 4** with Test MSE: 0.013126297716862124

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%2016.png)

**L2:**

| **Degree** | **Train MSE** | **Test MSE** | **Train Variance** | **Test Variance** | **Train Std Dev** | **Test Std Dev** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2268 | 0.1505 | 0.2240 | 0.1482 | 0.4733 | 0.3849 |
| 2 | 0.0680 | 0.0351 | 0.0672 | 0.0348 | 0.2591 | 0.1865 |
| 3 | 0.0634 | 0.0333 | 0.0625 | 0.0330 | 0.2500 | 0.1815 |
| 4 | 0.0337 | 0.0184 | 0.0329 | 0.0180 | 0.1814 | 0.1342 |
| 5 | 0.0329 | 0.0192 | 0.0321 | 0.0188 | 0.1792 | 0.1371 |
| 6 | 0.0204 | 0.0199 | 0.0197 | 0.0179 | 0.1403 | 0.1339 |
| 7 | 0.0187 | 0.0165 | 0.0181 | 0.0149 | 0.1347 | 0.1221 |
| 8 | 0.0177 | 0.0148 | 0.0172 | 0.0135 | 0.1313 | 0.1163 |
| 9 | 0.0170 | 0.0138 | 0.0165 | 0.0126 | 0.1286 | 0.1125 |
| 10 | 0.0162 | 0.0130 | 0.0158 | 0.0120 | 0.1256 | 0.1095 |
| 11 | 0.0188 | 0.0183 | 0.0186 | 0.0170 | 0.1365 | 0.1305 |
| 12 | 0.0234 | 0.0240 | 0.0228 | 0.0210 | 0.1509 | 0.1449 |
| 13 | 0.0283 | 0.0235 | 0.0275 | 0.0218 | 0.1658 | 0.1477 |
| 14 | 0.0269 | 0.0252 | 0.0261 | 0.0229 | 0.1616 | 0.1514 |
| 15 | 0.0276 | 0.0227 | 0.0268 | 0.0211 | 0.1636 | 0.1453 |
| 16 | 0.0260 | 0.0250 | 0.0255 | 0.0235 | 0.1596 | 0.1534 |
| 17 | 0.0248 | 0.0246 | 0.0244 | 0.0236 | 0.1564 | 0.1536 |
| 18 | 0.0208 | 0.0202 | 0.0207 | 0.0202 | 0.1440 | 0.1421 |
| 19 | 0.0265 | 0.0209 | 0.0257 | 0.0196 | 0.1604 | 0.1399 |
| 20 | 0.0262 | 0.0201 | 0.0254 | 0.0187 | 0.1594 | 0.1369 |

**Best Degree: 10** with Test MSE: 0.012999330221277772

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%207e737c2d9f71467e8251f781a0664165/image%2017.png)

From the MSE versus Degree Curve, we can see that L1 regularization does a better job for this dataset to reduce overfitting. However both the regularization techniques clearly do reduce overfitting in above case.

Note: LLM ChatGPT used for verifying and writing syntax of plotting the various graphs.