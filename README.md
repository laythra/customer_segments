## Machine Learning Nanodegree
### Customer Segments - Unsupervised Learning Project

In this project, I analyze a dataset containing data on various customers' annual spending amounts (reported in monetary units) of diverse product categories for internal structure.
One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with.

The project consists of several steps as follows:

#### Data Exploration
In the first step of this project, we explore the different features our dataset contains such as 'Fresh', 'Milk', 'Grocery' and more, 
each features represents a product category a customer could buy from, after that we do some necessary statistical analysis of our data
such as calculating the mean, std, the minimum and maximum value of each category, etc.
[!alt_text](https://github.com/laythra/customer_segments/blob/master/Images/stats.png)

#### Selecting Samples
In the next step, we pick three samples from our data and dig deeper into their details, we compare these three data points trying to predict
what kind of customer/establishment does each point represent.


#### Feature Relevance
What we try to do here is determine the least relevant feature(s) our dataset contains, this can be determined easily by fitting a linear regressor
to our dataset (excluding the feature we think it might irrelevant) and see how well our model is able to predict that excluded feature. I used the Decision Tree Regressor in this part and I excluded the **'Detergents_Paper'**, which means it will be the 
feature that our model will be trying to predict, after fitting the regressor and evaluating its performance we get a score of 0.74 which is pretty
decent, meaning this missing feature could be easily predicted and there's no need to keep on including in our dataset. Also, notice that
I used sklearn's **'train_test_split'** function to split our data in order to provide some never seen before data that our model could be evaluated using.

#### Visualize Feature Distributions
In order to get a better grasp of our dataset, a scatter matrix is constructed for each of the different 6 features our dataset contains,
The scatter matrix generally helped me understand if there are features that have some sort of correlation between them, we can also
see how the data for these features is distributed and so on.
[!alt_text](https://github.com/laythra/customer_segments/blob/master/Images/Visualization%20-%201.png)

### Preprocessing Step (my favorite step :bowtie:)
#### Feature Scaling
What we try to achieve in this step, is to make every feature have a distribution close to the normal distribution, since if our data
is not normally distributed then that means that the means and the median vary significantly, we can apply natural logarithm to do this job.
Here's how the scatter matrix of our data looks like after applying the natural logarithm, notice how it differs from the scatter matrix
representation above, it much closer to a normal distribution now and it's not skewed in any direction.
[!alt_text](https://github.com/laythra/customer_segments/blob/master/Images/Visualization%20-%202.png)

#### Outlier Detection
In this important preprocessing step, we try to detect the outliers our data contains, usually outliers negatively affect the performance of our model, and they can often skew the results.
I use **Tukey's Method for identifying outliers** and for each feature product I am able to see its detected outliers and drop
them out of my dataset.


### Feature Transformation
#### Implementing the PCA
In this part, we get to apply a really important concept called principal component analysis (often referred to as PCA) which 
calculates the dimensions the best maximize the variance.
We use the PCA function provided by the sklearn module and fit it to our "good_data" which had the necessary outliers removed and had been scaled 
to get closer to a normal distribution

[!alt_text](https://github.com/laythra/customer_segments/blob/master/Images/PCA.png)

Notice the picture above, that each dimension has an explained variance, which means how much variance within the data is explained by that dimension alone.

#### Dimensionality Reduction
Now it's time put that PCA we calculated into work, one of the PCA's main aims is to reduce the dimensionality of the data, reducing
the dimensionality of the data reduces the complexity of our problem heavily, however, we have to be careful because Dimensionality Reduction has its costs, because fewer dimensions means less variance, that why we could use the cumulative explained variance ratio calculated using
the PCA to determine how many dimensions are sufficient for our problem.


### Clustering
Let's dive into the core of the problem by solving it! I had two options to choose from, either to use **K-Means clustering algorithm** or to
use a **Gaussian Mixture Model clustering algorithm**, each has its advantages and disadvantages, after discussing each model
I ended up with picking the **Gaussian Mixture Model clustering algorithm**

#### The Implementation Part
In this part, i implemented the **Gaussian Mixture Model clustering algorithm** and I got to explore and play with it a little bit
by trying out a different number of clusters, I ended with the highest score when I tried 2 clusters only, the score I ended up with is: 0.42

#### Cluster Visualization
Here we simply visualized the results after choosing the optimal number of clustering (which is 2)
[!alt_text](https://github.com/laythra/customer_segments/blob/master/Images/Clusters.png)

### Conclusion

As for the rest of the project, i ended up investigating the ways that i can make use of the clustered data, and
Visualizing Underlying Distributions and discussing what each of the two clusters i ended up with could represent.


