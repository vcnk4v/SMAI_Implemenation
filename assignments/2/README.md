# Statistical Methods in Artificial Intelligence

Assignment 2

2022101104

## K-Means Clustering

### Determine the optimal number of clusters for 512-dimensions

The **elbow method** is a visual technique used to determine the optimal number of clusters for a given dataset. It involves plotting the Within Cluster Sum of Squares (WCSS) against the number of clusters to identify the most suitable value of k.

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%200.png" width="500" />

Analyzing the graph above, we can see that **k=6** appears to be a suitable elbow point. At this point, the steep descent nearly halts, followed by an almost linear decrease in the Within Cluster Sum of Squares (WCSS).

Hence **$k_{kmeans1}$ = 6**

# Gaussian Mixture Models

To evaluate the GMM class's performance, we apply it to a toy dataset from Kaggle containing 336 two-dimensional points.

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%201.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%201.png" width="500" />

## Determine Optimal Number of Clusters for 512 dimensions

The class does not function as expected, particularly when handling data with high dimensionality. It performs reasonably well up to 50 dimensions, but its performance significantly deteriorates when applied to datasets with higher dimensions, such as the 512-dimensional dataset in question.

The dataset consists of 512 dimensions and only 200 data points, which results in the covariance matrix being nearly singular. This causes the computed responsibilities to be extremely close to zero or negligible. Consequently, calculating the logarithm of these small values leads to an excessively negative log-likelihood.

To mitigate this issue, all divisions by zero in the E-step are replaced with ones, thereby avoiding any overflow due to division by zero.

Additionally, a safeguard is implemented to prevent poor initialization. If the log-likelihood after an iteration is lower than that of the previous iteration, the algorithm reverts to the original parameters (means, priors, covariances, and responsibilities).

This issue is not present in the scikit-learn implementation, as they use a more robust initialization process, such as the `estimate_gaussian_parameters` method, rather than random initialization. Furthermore, they employ several approximations to handle invalid divisions and overflows.

Hence, we use the sklearn class for this task.

```jsx
Reduced data shape: (200, 512)
Using own GMM: 216.13580737875427
Using sklearn GMM: 2721.568091395276
```

### Determining the Optimal Number of Clusters Using AIC and BIC

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%202.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%202.png" width="500" />

```jsx
{'optimal_bic_clusters': 1, 'optimal_aic_clusters': 1}
```

Hence, according to the plot, the optimal number of clusters is 1.

$k_{gmm1} = 1$

The reason for such peculiar result is the performance of GMM on high-dimensional data. GMMs perform so poorly that they end up doing no clustering at all.

# Dimensionality Reduction and Visualization

## Implementing PCA

Principal Component Analysis (PCA) is an unsupervised learning algorithm commonly used for dimensionality reduction. The PCA algorithm can be summarized in the following steps:

1. **Compute the Covariance Matrix:**
   - Calculate the covariance matrix of the data to understand how variables in the dataset vary with respect to each other. The covariance matrix provides insight into the scatter and relationships between the features.
2. **Calculate Eigenvalues and Eigenvectors:**
   - Perform eigenvalue decomposition on the covariance matrix to obtain its eigenvalues and corresponding eigenvectors. Eigenvalues indicate the amount of variance captured along each eigenvector, while eigenvectors represent the directions of maximum variance.
3. **Sort Eigenvalues:**
   - Sort the eigenvalues in descending order. Larger eigenvalues correspond to directions with higher variance in the dataset.
4. **Select Top k Eigenvalues and Eigenvectors:**
   - Choose the top k eigenvalues (where k is the number of principal components desired) and their corresponding eigenvectors. These eigenvectors define the new axes for the reduced-dimensional space.
5. **Transform the Data:**
   - Project the original data onto the space defined by the top k eigenvectors. This is done by taking the dot product of the data with the selected eigenvectors, resulting in a lower-dimensional representation of the data that retains the most significant variance.

By following these steps, PCA reduces the dimensionality of the dataset while preserving as much of the original variance as possible, making it easier to visualize and analyze the data.

![Transfromation to 2D](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/81f9be4e-c858-43e2-ab56-dba349200bcd.png)

Transfromation to 2D

![Tranformation to 3D](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/7b484cce-f3f5-40e5-860d-5e1d17ebd267.png)

Tranformation to 3D

**To identify what each of the new axes that are obtained from PCA represent**

The axes represent the principal components of the dataset. The first principal component captures the direction of maximum variance, while the second principal component captures the direction of second-highest variance, perpendicular to the first.

**To Estimate number of clusters from 2D and 3D plots**

Based on the 2D and 3D plots, the dataset appears to contain three distinct clusters.

Hence $k_2 = 3$.

# PCA + Clustering

## K-means Clustering Based on 2D Visualization

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%203.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%203.png" width="500" />

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%204.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%204.png" width="500" />

```jsx
No. of clusters: 3
Cluster assignments: [0 1 2 2 2 1 0 0 1 0 2 1 2 0 0 0 0 0 0 0 1 2 0 0 0 0 0 1 1 0 0 0 2 0 2 1 1
 0 0 0 1 0 1 1 2 0 1 0 0 0 0 0 0 1 0 0 2 1 2 1 2 2 2 1 0 1 0 2 1 0 2 1 1 1
 0 2 0 0 0 0 0 0 0 2 0 0 1 2 0 0 0 1 1 2 0 0 1 0 2 1 0 0 0 0 1 2 0 0 0 0 1
 1 0 1 0 2 1 0 0 0 0 0 0 0 2 1 0 1 1 1 1 1 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0
 0 0 0 0 0 1 2 0 1 0 0 2 0 2 0 0 0 2 0 0 1 2 0 1 2 1 2 0 0 0 1 0 0 0 0 0 0
 1 0 2 0 1 2 0 0 0 0 0 0 0 0 0]
Within-Cluster Sum of Squares (WCSS): 226.2565340734335

```

Performing kmeans on entire dataset with k2 = 3

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%205.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%205.png" width="500" />

```jsx
No. of clusters: 3
Cluster assignments: [1 1 2 2 2 1 2 1 1 2 1 1 2 2 2 2 2 1 1 2 1 2 2 1 2 2 2 1 1 2 2 2 2 2 1 1 1
 2 1 2 1 2 1 1 2 2 1 1 1 1 2 1 1 1 2 2 2 1 2 1 2 2 2 1 2 1 2 1 1 1 1 1 1 1
 2 2 2 2 2 1 0 2 2 2 2 1 1 2 2 2 1 1 1 2 1 0 1 2 2 1 2 2 2 2 1 2 0 2 2 2 1
 1 2 1 2 2 1 2 2 2 2 2 2 2 2 1 2 1 1 1 1 1 2 1 2 2 1 1 2 1 2 2 1 0 1 2 2 1
 2 1 2 2 2 1 2 2 1 1 2 2 1 2 2 2 2 2 2 2 1 0 2 1 2 1 2 2 2 1 1 2 1 2 2 2 1
 1 2 2 2 1 2 2 1 2 2 1 1 1 2 0]
Within-Cluster Sum of Squares (WCSS): 4227.5044323961065
```

## PCA + K-Means Clustering

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%206.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%206.png" width="500" />

Zoomed in Scree Plot till 50 dimensions:

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%207.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%207.png" width="500" />

### Determine Optimal Number of clusters for K-Means using Elbow Method

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%208.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%208.png" width="500" />

$k_{kmeans3} = 6$

```jsx
The WCSS of the reduced data is:  350.35655857631616
Cluster assignments: [1 1 2 5 5 4 1 1 4 2 5 4 5 0 0 3 0 1 1 1 1 2 2 1 2 2 2 4 4 2 2 0 3 0 5 1 4
 0 1 2 4 0 4 1 5 1 1 1 1 1 2 1 1 1 3 3 5 1 5 4 5 3 2 1 1 4 3 5 4 1 5 1 1 4
 0 5 3 0 0 1 1 3 3 5 3 1 4 5 1 2 1 4 1 2 1 2 4 0 5 1 0 2 3 3 4 3 0 3 2 3 4
 4 0 4 0 5 4 3 2 1 3 3 3 2 5 4 3 4 4 4 4 4 2 4 2 3 1 0 0 4 0 2 1 1 0 0 0 3
 1 1 2 3 1 4 3 3 1 1 3 5 1 5 2 2 2 5 0 3 4 3 3 4 2 4 5 3 1 0 4 0 1 3 2 2 1
 4 3 5 0 4 2 1 1 3 0 1 1 1 3 0]
```

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%209.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%209.png" width="500" />

## GMM Clustering Based on 2D Visualization

On dataset reduced to 2 dims:

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2010.png) -->

<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2010.png" width="500" />

```jsx
Log likelihood: -2.9182661686429245
```

On dataset of 512 dims:

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2011.png) -->

<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2011.png" width="500" />

```jsx
Log likelihood: 248.60171666035393
```

## PCA + GMMs

### Determine Optimal Clusters Using AIC and BIC on reduced dataset

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2012.png) -->

<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2012.png" width="500" />

```jsx
{'optimal_bic_clusters': 3, 'optimal_aic_clusters': 8}
```

$k_{gmm3} = 3$

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2013.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2013.png" width="500" />

```jsx
Log Likelihood: -5.7906839542222475
```

# Cluster Analysis

## K-Means Cluster Analysis

$k_{kmeans1} =$ 6: This is the optimal number of clusters for the 512D data, determined using WCSS and the Elbow plot.

$k_{2} =$ 3: This result is derived from 2D visualization of the data and observation.

$k_{kmeans1} =$ 6: This is the optimal number of clusters for the 5D data, determined using WCSS and the Elbow plot.

To compare clustering results effectively, we need to visualize the dataset in 2D, using labels obtained from the respective clustering methods in each case.

**For $k_{kmeans1}$:**

```python
Cluster 0: saturn, paintbrush

Cluster 1: needle, eraser, table, brick, brush, mug, spoon, passport, stairs,
microwave, notebook, cigarette, microphone, baseball, bench, bucket, feet, boat,
basket, flute, scissor, laptop, door, calendar, chair, ladder, finger, candle,
clock, oven, calculator, tree, envelope, skate, hammer, toothbrush, screwdriver,
fingerprints, teaspoon, dustbin, ambulance, pizza, television, throne, camera,
sword, loudspeaker, telephone, stove, knife, toothpaste, basketball, wheel,
bicycle, toaster, comb, shoe, walk, keyboard, fork, radio, truck, suitcase

Cluster 2: rose, carrot, bear, spider, shark, grass, giraffe, lizard, feather,
frog, puppet, lake, monkey, starfish, plant, puppy, fish, grape, mouse, goldfish,
bird, spiderman, snake, airplane, dragonfly, butterfly, arrow, crocodile

Cluster 3: fishing, forest, gym, roof, bed, knit, sweater, jacket, fruit,
badminton, igloo, pillow, length, rain, tent, parachute, lantern, rainy,
windmill

Cluster 4: drive, sing, listen, dive, flame, sit, knock, exit, smile, bullet,
bury, download, eat, postcard, hard, bend, fight, call, fly, face, climb, kneel,
scream, kiss, selfie, catch, hit, paint, far, dig, cry, run, clap, pull, sleep,
hollow, clean, sad, empty, slide, drink, draw, pray, arrest, email, buy, burn,
fire, close, angry, lazy, scary, hang, book, tattoo, earth, tank, enter, key,
swim, zip, happy, loud, love, cook, recycle, cut, sunny

Cluster 5: deer, panda, ape, helicopter, cat, rifle, cow, pencil, van, sun, pear,
peacock, ant, bee, beetle, tomato, car, elephant, pant, potato
```

**Cluster 1: Everyday Objects and Tools**

Includes commonly used items like household objects (microwave, chair), personal items (passport, camera), and tools (screwdriver, hammer). Represents tangible objects used in daily activities.

**Cluster 2: Animals, Insects, and Nature**

Comprises animals (giraffe, bear), insects (dragonfly, butterfly), and natural elements (lake, grass). Focuses on living creatures and nature, distinct from other clusters of inanimate objects or actions.

**Cluster 3: Outdoor Activities and Weather**

Contains outdoor activities (fishing, camping), weather terms (rainy, windy), and sports (badminton, gym). Highlights outdoor life and scenarios involving nature and movement.

**Cluster 4: Actions and Emotions**

Centers on human activities (drive, sing), emotions (sad, happy), and abstract concepts (love, sunny). Reflects human experiences, actions, and emotional states.

**Cluster 5: Animals, Vehicles, and Tools**

Includes a mix of animals (dog, horse), vehicles (car, bicycle), and tools (wrench, shovel). Represents a diverse set of physical entities, blending living creatures with man-made objects.

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2014.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2014.png" width="500" />

**For $k_{2}$:**

```python
Cluster 0: drive, rose, dive, helicopter, needle, eraser, table, carrot, exit,
brick, fishing, spider, bullet, shark, grass, giraffe, forest, lizard, brush,
feather, spoon, bend, frog, puppet, lake, climb, kneel, scream, monkey, kiss,
passport, roof, stairs, plant, microwave, notebook, knit, sweater, cigarette,
microphone, baseball, hollow, jacket, bench, bucket, puppy, boat, pear, basket,
fish, saturn, flute, scissor, grape, laptop, door, badminton, chair, mouse,
ladder, finger, candle, igloo, goldfish, bird, clock, oven, calculator, spiderman,
pillow, tree, beetle, envelope, skate, hammer, toothbrush, screwdriver, snake,
tattoo, fingerprints, teaspoon, length, dustbin, rain, tank, airplane, ambulance,
pizza, throne, swim, tent, zip, dragonfly, parachute, butterfly, sword, loudspeaker,
telephone, stove, rainy, knife, toothpaste, basketball, wheel, bicycle, windmill,
arrow, toaster, comb, shoe, walk, keyboard, fork, sunny, radio, truck, suitcase,
paintbrush

Cluster 1: sing, listen, flame, knock, smile, bury, download, postcard, hard, fight,
call, fly, face, selfie, catch, hit, paint, far, cry, clap, pull, sleep, clean, sad,
empty, slide, drink, draw, pray, arrest, email, buy, burn, fire, close, angry,
lazy, scary, hang, book, earth, enter, key, happy, loud, love, cook, recycle, cut

Cluster 2: deer, panda, ape, sit, cat, bear, mug, eat, gym, rifle, cow, pencil,
bed, starfish, dig, run, van, sun, feet, peacock, fruit, calendar, ant, bee,
television, camera, tomato, car, lantern, elephant, pant, potato, crocodile
```

**Cluster 0**: A diverse category lacking a clear theme. It encompasses actions, objects, animals, and natural elements. The cluster seems to group words based on loose associations, potentially linking outdoor activities with animals and tools, but it lacks overall coherence.

**Cluster 1**: A highly cohesive cluster of **actions**, **emotions**, and **abstract concepts**. This group revolves around human experiences, emotional states, and behaviors, distinguishing itself from clusters that focus more on animals and objects.

**Cluster 2**: A blend of animals and everyday objects, with some actions mixed in. The objects are typically domestic or utility-related, while the animals span both wild and domestic species. This cluster highlights living beings and common items encountered in daily life.

These clusters lack the coherence of previous clustering results due to the dimensional reduction from 512D to 2D before clustering—essentially, visual clustering based on a 2D projection. This drastic reduction in dimensions leads to significant information loss, resulting in the formation of some unrelated clusters.

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2015.png) -->

<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2015.png" width="500" />

**For $k_{kmeans3}$:**

```python
Cluster 0: needle, eraser, carrot, brush, feather, spoon, puppet, knit, cigarette,
microphone, flute, scissor, badminton, finger, candle, hammer, toothbrush,
screwdriver, fingerprints, teaspoon, length, sword, knife, toothpaste, comb, fork,
paintbrush

Cluster 1: drive, sing, rose, dive, exit, brick, fishing, smile, bullet, postcard,
bend, fly, lake, face, climb, kneel, scream, kiss, passport, selfie, catch, paint,
plant, notebook, clap, pull, baseball, hollow, puppy, boat, basket, empty, fish,
drink, bird, skate, tattoo, earth, rain, tank, pizza, key, swim, zip, rainy,
basketball, arrow, shoe, walk, sunny, radio, truck

Cluster 2: deer, helicopter, bear, spider, shark, grass, giraffe, forest, lizard,
frog, monkey, starfish, pear, peacock, saturn, grape, mouse, goldfish, spiderman,
tree, beetle, snake, airplane, dragonfly, parachute, butterfly, elephant, bicycle,
windmill, crocodile

Cluster 3: table, mug, roof, stairs, bed, microwave, sweater, jacket, bench, bucket,
laptop, door, calendar, chair, ladder, igloo, clock, oven, calculator, pillow,
envelope, dustbin, ambulance, television, throne, tent, loudspeaker, lantern,
telephone, stove, wheel, toaster, keyboard, suitcase

Cluster 4: listen, flame, knock, bury, download, hard, fight, call, hit, far, cry,
sleep, clean, sad, slide, draw, pray, arrest, email, buy, burn, fire, close, angry,
lazy, scary, hang, book, enter, happy, loud, love, cook, recycle, cut

Cluster 5: panda, ape, sit, cat, eat, gym, rifle, cow, pencil, dig, run, van, sun,
feet, fruit, ant, bee, camera, tomato, car, pant, potato
```

**Cluster 0**: Focuses on **small tools, instruments, and household items**. This group primarily consists of **functional, tangible objects**.

**Cluster 1**: Combines **actions, emotions, and outdoor activities**. It emphasizes **motion, action, and emotions**, while incorporating some natural elements.

**Cluster 2**: Centers on **animals, nature, and adventure**, grouping wildlife and exploration themes.

**Cluster 3**: A domestic cluster featuring **furniture, appliances, and structural items** typically found in homes or public spaces.

**Cluster 4**: A cohesive group focused on **actions, emotions, and states of being**, representing personal experiences and feelings.

**Cluster 5**: Blends **animals, actions, and everyday objects**, combining the natural world with practical, day-to-day activities.

<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2016.png" width="500" />

$k_{kmeans} = 6$

**Conclusion**

kmeans1 i.e., clustering on 512-dimensional data, yields the best results in terms of more coherent clusters with the least overlap. We can observe that kmeans3 also produces similar clusters but with slightly more overlap. This is due to the loss of some semantic features during dimensionality reduction, which might be crucial in differentiating words between clusters. For instance, some animals and actions tend to mix in kmeans3, which doesn't occur in kmeans1. This could result from dropping or assigning lower importance to dimensions that distinguish animals from actions during dimensionality reduction.

## GMM Cluster Analysis

$k_{gmm1} =$ 1 : GMM on 512 dimensional data

$k_2 =$ 3: This is optimal number of clusters from 2D visualization of data

$k_{gmm3}=$ 3: This is optimal number of clusters for 5D data using AIC or BIC

**For $k_{gmm1}:$**

```python
Cluster 0: drive, sing, deer, panda, ape, listen, rose, dive, flame, helicopter,
sit, knock, cat, needle, eraser, table, carrot, exit, brick, fishing, smile,
bear, spider, bullet, shark, grass, giraffe, bury, download, forest, lizard,
brush, mug, feather, eat, postcard, hard, spoon, bend, frog, fight, puppet,
call, fly, gym, lake, face, climb, kneel, scream, monkey, kiss, passport,
selfie, roof, stairs, rifle, catch, cow, hit, pencil, bed, starfish, paint,
plant, far, microwave, dig, cry, notebook, run, clap, pull, sleep, knit, van,
sweater, cigarette, microphone, baseball, hollow, jacket, bench, sun, bucket,
puppy, clean, feet, boat, pear, basket, sad, empty, peacock, fish, saturn, slide,
flute, fruit, drink, scissor, grape, laptop, door, draw, calendar, badminton,
chair, mouse, ladder, pray, arrest, finger, email, candle, ant, buy, igloo,
goldfish, bird, clock, oven, calculator, spiderman, bee, burn, pillow, fire,
close, angry, lazy, scary, tree, hang, beetle, envelope, skate, hammer,
toothbrush, book, screwdriver, snake, tattoo, earth, fingerprints, teaspoon,
length, dustbin, rain, tank, airplane, ambulance, pizza, enter, television,
throne, key, swim, tent, camera, zip, tomato, dragonfly, parachute, butterfly,
car, sword, loudspeaker, happy, lantern, telephone, loud, elephant, love, pant,
stove, rainy, knife, cook, toothpaste, basketball, wheel, bicycle, windmill,
arrow, recycle, toaster, potato, comb, cut, crocodile, shoe, walk, keyboard,
fork, sunny, radio, truck, suitcase, paintbrush
```

This clustering attempt fails to produce meaningful groupings, rendering any analysis futile.

<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2017.png" width="500" />

**For $k_2:$**

```python
Cluster 0: panda, ape, sit, cat, eat, gym, rifle, cow, pencil, bed, starfish,
dig, run, van, sun, sad, peacock, ant, buy, bee, camera, tomato, car, lantern,
elephant, pant, potato, crocodile

Cluster 1: drive, deer, rose, dive, helicopter, needle, eraser, table, carrot,
exit, brick, fishing, smile, bear, spider, bullet, shark, grass, giraffe, forest,
lizard, brush, mug, feather, postcard, spoon, bend, frog, puppet, fly, lake,
face, climb, kneel, scream, monkey, kiss, passport, selfie, roof, stairs, catch,
paint, plant, microwave, notebook, clap, pull, knit, sweater, cigarette,
microphone, baseball, hollow, jacket, bench, bucket, puppy, feet, boat, pear,
basket, empty, fish, saturn, flute, fruit, scissor, grape, laptop, door,
calendar, badminton, chair, mouse, ladder, arrest, finger, candle, igloo,
goldfish, bird, clock, oven, calculator, spiderman, pillow, angry, tree,
beetle, envelope, skate, hammer, toothbrush, screwdriver, snake, tattoo, earth,
fingerprints, teaspoon, length, dustbin, rain, tank, airplane, ambulance, pizza,
television, throne, key, swim, tent, zip, dragonfly, parachute, butterfly, sword,
loudspeaker, telephone, stove, rainy, knife, toothpaste, basketball, wheel,
bicycle, windmill, arrow, toaster, comb, cut, shoe, walk, keyboard, fork, sunny,
radio, truck, suitcase, paintbrush

Cluster 2: sing, listen, flame, knock, bury, download, hard, fight, call, hit,
far, cry, sleep, clean, slide, drink, draw, pray, email, burn, fire, close, lazy,
scary, hang, book, enter, happy, loud, love, cook, recycle
```

**Cluster 0:** This cluster primarily groups **animals**, **objects**, and **basic actions**. It includes tangible items and living creatures, with some emphasis on everyday activities like sitting and eating, as well as settings like a gym. Despite its diversity, it maintains a common theme around **physical entities and simple actions**.

**Cluster 1:** This is a **broad cluster** that encompasses a wide range of elements, including **everyday objects** (tables, ladders, laptops), **animals** (frogs, monkeys, sharks), as well as **emotions** and **actions** (climb, scream, angry, pull). While diverse, this cluster loosely centers around **common objects, activities, and places** found in daily life, along with some emotional states. It reflects a general theme of **life experiences** where interactions with the environment and objects are central.

**Cluster 2:** This cluster has a more **abstract** and **action-oriented** focus, with words related to **emotional states** (happy, love, scary, lazy) and **actions** (fight, cry, draw, cook, recycle). This is the most cohesive cluster, concentrating on **abstract concepts** like emotions, human interactions, and states of being, without mixing in physical items or animals. It clearly captures **how people feel and what they do**.

**Conclusion:**

**Cluster 2** is the most consistent and semantically meaningful, focusing on emotions and actions. **Cluster 1** follows, representing a broad mix of everyday experiences. **Cluster 0** is less cohesive due to its diversity but still has a recognizable structure around physical objects and animals.

This reasoning aligns with the results seen in K-means clustering for k=2. Since the clustering is performed on a 2D projection of the data, many features that distinguish semantically different words might have been lost, reducing the ability of the GMM to differentiate between categories like animals and objects. However, it still effectively separates more distinct categories like animals and emotions.

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2018.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2018.png" width="500" />

**For $k_{gmm3}:$**

```python
Cluster 0: deer, panda, ape, rose, cat, needle, eraser, carrot, bear, spider,
bullet, shark, giraffe, lizard, brush, feather, spoon, frog, puppet, fly, scream,
monkey, rifle, cow, pencil, starfish, knit, cigarette, microphone, sun, puppy,
pear, peacock, saturn, flute, scissor, grape, mouse, finger, candle, ant,
goldfish, bird, spiderman, bee, beetle, hammer, toothbrush, screwdriver, snake,
fingerprints, teaspoon, length, zip, tomato, dragonfly, butterfly, sword, elephant,
knife, toothpaste, arrow, comb, crocodile, fork, paintbrush

Cluster 1: drive, sing, dive, helicopter, sit, table, exit, brick, fishing, smile,
grass, forest, mug, postcard, call, gym, lake, climb, kneel, kiss, passport,
selfie, roof, stairs, catch, bed, paint, plant, microwave, notebook, van, sweater,
baseball, jacket, bench, bucket, feet, boat, basket, fish, slide, fruit, drink,
laptop, door, calendar, badminton, chair, ladder, email, igloo, clock, oven,
calculator, pillow, tree, envelope, skate, book, tattoo, earth, dustbin, rain,
tank, airplane, ambulance, pizza, television, throne, swim, tent, camera,
parachute, car, loudspeaker, lantern, telephone, stove, rainy, basketball, wheel,
bicycle, windmill, toaster, shoe, walk, keyboard, sunny, radio, truck, suitcase

Cluster 2: listen, flame, knock, bury, download, eat, hard, bend, fight, face,
hit, far, dig, cry, run, clap, pull, sleep, hollow, clean, sad, empty, draw,
pray, arrest, buy, burn, fire, close, angry, lazy, scary, hang, enter, key,
happy, loud, love, pant, cook, recycle, potato, cut
```

**Cluster 0:** This cluster primarily consists of **animals**, **tools/objects**, and some **nature-related items**. It brings together living entities like animals with man-made tools and smaller objects, suggesting a broad grouping of tangible elements. These items can be categorized as either **natural elements** (animals, nature) or **man-made objects**. A few **actions** like "fly" and "scream" add a sense of interaction between living things and their surroundings.

**Cluster 1:** This cluster revolves around **everyday life settings and interactions**, featuring **places** (lake, roof, tent), **objects** (table, chair, ladder), and **human activities** (sing, smile, catch, paint). It reflects interactions with the environment and daily routines, where people use tools, move through various settings, and engage in familiar activities.

**Cluster 2:** This cluster focuses on **actions and emotions**, grouping together **verbs** that describe physical actions (run, eat, dig, clap) and **emotional states** (happy, sad, angry, loud). It is centered on **human behavior** and **emotional experiences**, forming a coherent group of **abstract, emotional, and interaction-based words**.

<!-- ![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2019.png) -->
<img src="Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2019.png" width="500" />

**Conclusion:**

The clustering results from **gmm3** are the most meaningful among **gmm1**, **k2**, and **gmm3**.

- **kgmm1** struggles due to GMM's limitations in handling high-dimensional data with nearly singular covariance matrices.
- **k2** performs GMM on a 2D projection of the 512-dimensional data, resulting in less impressive clusters due to the loss of many distinguishing features.
- **kgmm3** provides the best results as it reduces the data to 5 dimensions, capturing most of the variance according to the scree plot. AIC/BIC analysis indicates that 3 clusters offer the optimal balance, suggesting that the data is most likely drawn from 3 Gaussian distributions.

**$k_{gmm} = 3$**

Thus, **gmm with 3 clusters** yields fairly interpretable and cohesive clustering results.

## Compare K-Means and GMMs

Using the above results for k = 6 in Kmeans and k = 3 in GMM,

K-means (k=3)

### Similarity Within Clusters

- **K-means Clustering (k=6):** The clusters seem to group words based on their general category or function. For example, Cluster 5 includes actions and objects related to physical activities (e.g., "run," "dig," "eat"), while Cluster 1 contains a mix of everyday objects and actions (e.g., "mug," "paint," "smile"). This might indicate that K-means groups words by similar usage contexts or features, but the granularity might vary.
- **GMM Clustering (k=3):** The clusters appear to focus on more specific features. For instance, Cluster 2 contains words related to emotions and states (e.g., "sad," "happy," "angry"), while Cluster 0 groups objects and tools (e.g., "needle," "toothpaste," "scissors"). This might suggest that GMM identifies more nuanced patterns in the data, grouping words with similar abstract features or functions.

### Separation Between Clusters

- **K-means Clustering:** K-means aims to minimize within-cluster variance, which often leads to well-separated clusters. However, it might sometimes force clusters into spherical shapes, which may not always fit the data well if the true cluster shapes are more complex.
- **GMM Clustering:** GMM provides a probabilistic approach that can handle more complex cluster shapes (elliptical) and overlaps. This might lead to better separation in cases where clusters are not well-separated or have varying densities.

### General Observations

- **K-means:** Tends to produce clusters that are more interpretable based on the mean of the data points, which can be beneficial for clear-cut categories but might miss subtler distinctions.
- **GMM:** Can capture more complex relationships and overlaps, which might be useful if the data inherently forms overlapping clusters or if there's variability within clusters.
- **Conclusion:** K-means produces more distinct groupings, effectively separating objects, animals, and actions/emotions. GMM clusters exhibit greater overlap, potentially capturing subtle similarities but yielding less interpretable results. K-means demonstrates clearer within-cluster similarity, while GMM suggests more nuanced relationships between items..

# Hierarchical Clustering

## Perform hierarchical clustering using hc.linkage and hc.dendogram

The `hc.linkage` function returns an (n−1)×4(n-1) \times 4(n−1)×4 matrix, `Z`, where `n` is the number of original data points. Each row of the matrix represents a step in the hierarchical clustering process:

- **`Z[i, 0]`** and **`Z[i, 1]`** contain the indices of the clusters that are merged in the i-th iteration. If an index is less than n, it corresponds to one of the original observations. Otherwise, it references a newly formed cluster from previous iterations.
- **`Z[i, 2]`** indicates the distance between the clusters being merged.
- **`Z[i, 3]`** specifies the number of original data points in the newly formed cluster.

This matrix encodes the hierarchical structure of the clustering, which can then be used to construct dendrograms for visualization.

## Analyzing Different Distance Metrics and Linkage Methods

Linkage methods are methods used to compute the distance between clusters for hierarchical clustering be it between original clusters for that iteration or newly formed clusters from unused clusters in that iteration.

Distance metrics are used to compute pairwise distance between observations in our y matrix. The common distance metrics are `euclidean, manhattan and cosine`.

Different linkage methods are as follows: `single, complete, average, centroid, median, ward, weighted`

centroid, median and ward linkage methods only work with euclidean distance metric.

The plots for each metric and linkage method;
| ![hc_cityblock_average.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_cityblock_average.png) | ![hc_cityblock_complete.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_cityblock_complete.png) | ![hc_cityblock_single.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_cityblock_single.png) |
|---|---|---|
| ![hc_cityblock_weighted.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_cityblock_weighted.png) | ![hc_cosine_average.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_cosine_average.png) | ![hc_cosine_complete.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_cosine_complete.png) |
| ![hc_cosine_single.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_cosine_single.png) | ![hc_cosine_weighted.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_cosine_weighted.png) | ![hc_euclidean_average.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_euclidean_average.png) |
| ![hc_euclidean_centroid.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_euclidean_centroid.png) | ![hc_euclidean_complete.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_euclidean_complete.png) | ![hc_euclidean_median.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_euclidean_median.png) |
| ![hc_euclidean_single.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_euclidean_single.png) | ![hc_euclidean_ward.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_euclidean_ward.png) | ![hc_euclidean_weighted.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/hc_euclidean_weighted.png) |

### Observations:

**Single Linkage:**

- Single linkage can create long, chain-like clusters, especially with the cosine metric. This can lead to poorly defined groups because distant points get linked together. So, single linkage might not be the best choice.

**Complete Linkage:**

- Complete linkage generally makes clearer, more distinct clusters. It avoids long chains and works well with Euclidean and cityblock metrics. The cosine metric also works with complete linkage but may create tightly packed clusters at low distances or high similarities.

**Average Linkage:**

- Average linkage balances single and complete linkage. It usually produces well-separated clusters and keeps a good hierarchical structure. However, it can still have chaining issues with the cosine metric.

**Weighted Linkage:**

- Weighted linkage is similar to average linkage but less effective at separating clusters. With the cityblock metric, it can make clusters that are closely packed and harder to distinguish.

**Centroid Linkage:**

- Centroid linkage with Euclidean distance gives decent results but doesn’t offer much advantage over average or complete linkage. The clusters might be less balanced and sometimes merged together.

**Median Linkage:**

- Median linkage is like centroid linkage. It creates clusters with a similar structure but can result in more mixed or overlapping clusters. It may not be the best for making well-separated groups.

**Ward Linkage:**

- Ward’s method with Euclidean distance is great for creating distinct and well-separated clusters. It minimizes variance within clusters and is often recommended for hierarchical clustering. The clusters are usually clean and well-defined.

**In summary, _Ward Linkage_ is the best method for hierarchical clustering because it creates clear and balanced clusters.**

## Comparing With KMeans and GMM Clustering

$k_{gmm} = 3$

$k_{kmeans} = 6$

```python
Cluster 1: sing, listen, dive, flame, knock, exit, brick, smile, bury, download, hard, bend, fight, face, scream, kiss, selfie, catch, hit, paint, far, cry, sleep, hollow, clean, sad, empty, slide, drink, door, draw, pray, arrest, buy, burn, fire, close, angry, lazy, scary, hang, tattoo, earth, enter, key, swim, happy, loud, love, cook, cut

Cluster 2: deer, panda, ape, sit, cat, eraser, carrot, bear, spider, shark, grass, giraffe, forest, lizard, feather, eat, frog, puppet, fly, gym, kneel, monkey, cow, pencil, starfish, plant, dig, run, clap, pull, sun, puppy, feet, pear, peacock, fish, fruit, grape, finger, ant, goldfish, bird, spiderman, bee, tree, beetle, snake, fingerprints, rain, zip, tomato, dragonfly, butterfly, elephant, pant, rainy, potato, crocodile, shoe, sunny

Cluster 3: drive, rose, helicopter, needle, table, fishing, bullet, brush, mug, postcard, spoon, call, lake, climb, passport, roof, stairs, rifle, bed, microwave, notebook, knit, van, sweater, cigarette, microphone, baseball, jacket, bench, bucket, boat, basket, saturn, flute, scissor, laptop, calendar, badminton, chair, mouse, ladder, email, candle, igloo, clock, oven, calculator, pillow, envelope, skate, hammer, toothbrush, book, screwdriver, teaspoon, length, dustbin, tank, airplane, ambulance, pizza, television, throne, tent, camera, parachute, car, sword, loudspeaker, lantern, telephone, stove, knife, toothpaste, basketball, wheel, bicycle, windmill, arrow, recycle, toaster, comb, walk, keyboard, fork, radio, truck, suitcase, paintbrush

```

- **Cluster 1 in both methods** shares significant overlap in actions and some objects, though GMM Cluster 1 includes more physical objects (e.g., _car_, _television_) than Hierarchical Cluster 1.
- **Cluster 2 in hierarchical clustering and Cluster 0 in GMM** both focus on animals, nature, and objects, with greater alignment in themes such as animals and insects.
- **Cluster 3 in hierarchical clustering and Cluster 2 in GMM** differ significantly: Hierarchical Cluster 3 is predominantly object-based, while GMM Cluster 2 combines objects with actions and emotions.

Overall, **GMM clustering** tends to blend objects with actions and emotions more than **hierarchical clustering**, which exhibits a clearer separation between word types such as actions, emotions, and objects.

```python
Cluster 1: sing, listen, dive, flame, knock, exit, brick, smile, bury, download, hard, bend, fight, face, scream, kiss, selfie, catch, hit, paint, far, cry, sleep, hollow, clean, sad, empty, slide, drink, door, draw, pray, arrest, buy, burn, fire, close, angry, lazy, scary, hang, tattoo, earth, enter, key, swim, happy, loud, love, cook, cut

Cluster 2: deer, spider, shark, giraffe, lizard, feather, frog, fly, starfish, peacock, fish, ant, goldfish, bird, spiderman, bee, beetle, snake, dragonfly, butterfly, crocodile

Cluster 3: panda, ape, sit, cat, eraser, carrot, bear, grass, forest, eat, puppet, gym, kneel, monkey, cow, pencil, plant, dig, run, clap, pull, sun, puppy, feet, pear, fruit, grape, finger, tree, fingerprints, rain, zip, tomato, elephant, pant, rainy, potato, shoe, sunny

Cluster 4: brush, spoon, scissor, hammer, toothbrush, screwdriver, teaspoon, length, sword, knife, toothpaste, comb, fork, paintbrush

Cluster 5: postcard, call, passport, microwave, notebook, microphone, laptop, calendar, email, oven, calculator, envelope, book, dustbin, television, camera, loudspeaker, telephone, stove, recycle, toaster, keyboard, radio, suitcase

Cluster 6: drive, rose, helicopter, needle, table, fishing, bullet, mug, lake, climb, roof, stairs, rifle, bed, knit, van, sweater, cigarette, baseball, jacket, bench, bucket, boat, basket, saturn, flute, badminton, chair, mouse, ladder, candle, igloo, clock, pillow, skate, tank, airplane, ambulance, pizza, throne, tent, parachute, car, lantern, basketball, wheel, bicycle, windmill, arrow, walk, truck

```

- **Cluster 1 in hierarchical** largely matches **Cluster 4 in K-means**, focusing on actions and emotions.
- **Cluster 2 in both methods** shows strong alignment, focusing on animals and nature.
- **Cluster 4 in hierarchical** and **Cluster 1 in K-means** align well in grouping small tools and household objects.
- **Cluster 3** shows some differences, with **hierarchical** being more animal-focused and **K-means** grouping a wider range of activities and objects.

Overall, there’s significant alignment in clusters focused on actions, objects, and animals between the two methods, though K-means seems to distribute actions and objects more widely.

We can also plot these clusters by reducing the dataset to 2 dimensions using PCA and using the labels obtained from hierarchical clustering.

# Nearest Neighbor Search

## PCA + KNN

### Use Scree Plot To Determine Optimal Number of Dimensions For Reduction

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2020.png)

Based on the graph, the optimal number of dimensions for reducing the Spotify dataset appears to be 4. However, we could also consider reducing it to 10 dimensions to capture slightly more variance in the data without significantly compromising accuracy.

### Use Best KNN Model on Reduced Dataset

| No. Dimensions | Accuracy            | Precision           | Recall              | F1 Score Macro      | F1 Score Micro      |
| -------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| 10             | 0.22736842105263158 | 0.21654777187253502 | 0.22674601907848696 | 0.22152958699048736 | 0.2058823529411765  |
| 4              | 0.1281578947368421  | 0.1192271025637618  | 0.12809278146645794 | 0.12350103796517083 | 0.03669724770642201 |
| 512            | 0.25649122807017544 | 0.24662496307188722 | 0.2551112319960177  | 0.2507963299786509  | 0.22564102564102564 |

From the above results, we observe that the accuracy for 4D data is approximately 12.81%, which improves to 22.73% for 10D data and reaches 25.64% for the full dataset. This indicates that as dimensionality decreases, accuracy tends to decline due to the loss of information. Therefore, in practical scenarios, there is a tradeoff between accuracy and performance. Additionally, precision, recall, and F1 score also show consistent improvement as dimensionality increases.

```python
Time for reduced dataset(4D):  78.445
Time for reduced dataset(10D):  90.776
Time for complete dataset:  96.512
```

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%204368671dcc8e448d91e0384c14c55c25/image%2021.png)

As shown in the graph above, there is only a slight change in inference times as dimensionality increases. This is due to the optimized nature of the algorithm. In optimizedKNN, we compute distances between data points across rows—essentially processing multiple features simultaneously in parallel. Consequently, changes in the number of features don't significantly increase inference time, thanks to vectorization. In contrast, the change would be more pronounced in NaiveKNN, where each feature is computed sequentially.
