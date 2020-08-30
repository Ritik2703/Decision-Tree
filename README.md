# Decision-Tree
# hare krishna
# Radhe Radhe

# Decision Trees in Machine Learning


A tree has many analogies in real life, and turns out that it has influenced a wide area of machine learning, covering both classification and regression. In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making. As the name goes, it uses a tree-like model of decisions. Though a commonly used tool in data mining for deriving a strategy to reach a particular goal, its also widely used in machine learning, which will be the main focus of this article.

# How can an algorithm be represented as a tree?
 
For this let’s consider a very basic example that uses titanic data set for predicting whether a passenger will survive or not. Below model uses 3 features/attributes/columns from the data set, namely sex, age and sibsp (number of spouses or children along).
Image for post


A decision tree is drawn upside down with its root at the top. In the image on the left, the bold text in black represents a condition/internal node, based on which the tree splits into branches/ edges. The end of the branch that doesn’t split anymore is the decision/leaf, in this case, whether the passenger died or survived, represented as red and green text respectively.

Although, a real dataset will have a lot more features and this will just be a branch in a much bigger tree, but you can’t ignore the simplicity of this algorithm. The feature importance is clear and relations can be viewed easily. This methodology is more commonly known as learning decision tree from data and above tree is called Classification tree as the target is to classify passenger as survived or died. Regression trees are represented in the same manner, just they predict continuous values like price of a house. In general, Decision Tree algorithms are referred to as CART or Classification and Regression Trees.
So, what is actually going on in the background? Growing a tree involves deciding on which features to choose and what conditions to use for splitting, along with knowing when to stop. As a tree generally grows arbitrarily, you will need to trim it down for it to look beautiful. Lets start with a common technique used for splitting.

# Recursive Binary Splitting
In this procedure all the features are considered and different split points are tried and tested using a cost function. The split with the best cost (or lowest cost) is selected.
Consider the earlier example of tree learned from titanic dataset. In the first split or the root, all attributes/features are considered and the training data is divided into groups based on this split. We have 3 features, so will have 3 candidate splits. Now we will calculate how much accuracy each split will cost us, using a function. The split that costs least is chosen, which in our example is sex of the passenger. This algorithm is recursive in nature as the groups formed can be sub-divided using same strategy. Due to this procedure, this algorithm is also known as the greedy algorithm, as we have an excessive desire of lowering the cost. This makes the root node as best predictor/classifier.

# Cost of a split
Lets take a closer look at cost functions used for classification and regression. In both cases the cost functions try to find most homogeneous branches, or branches having groups with similar responses. This makes sense we can be more sure that a test data input will follow a certain path.

Regression : sum(y — prediction)²

Lets say, we are predicting the price of houses. Now the decision tree will start splitting by considering each feature in training data. The mean of responses of the training data inputs of particular group is considered as prediction for that group. The above function is applied to all data points and cost is calculated for all candidate splits. Again the split with lowest cost is chosen. Another cost function involves reduction of standard deviation, more about it can be found here.

# Classification : G = sum(pk * (1 — pk))
 
 A Gini score gives an idea of how good a split is by how mixed the response classes are in the groups created by the split. Here, pk is proportion of same class inputs present in a particular group. A perfect class purity occurs when a group contains all inputs from the same class, in which case pk is either 1 or 0 and G = 0, where as a node having a 50–50 split of classes in a group has the worst purity, so for a binary classification it will have pk = 0.5 and G = 0.5.

# When to stop splitting?

You might ask when to stop growing a tree? As a problem usually has a large set of features, it results in large number of split, which in turn gives a huge tree. Such trees are complex and can lead to overfitting. So, we need to know when to stop? One way of doing this is to set a minimum number of training inputs to use on each leaf. For example we can use a minimum of 10 passengers to reach a decision(died or survived), and ignore any leaf that takes less than 10 passengers. Another way is to set maximum depth of your model. Maximum depth refers to the the length of the longest path from a root to a leaf.
 
# Pruning

The performance of a tree can be further increased by pruning. It involves removing the branches that make use of features having low importance. This way, we reduce the complexity of tree, and thus increasing its predictive power by reducing overfitting.
Pruning can start at either root or the leaves. The simplest method of pruning starts at leaves and removes each node with most popular class in that leaf, this change is kept if it doesn't deteriorate accuracy. Its also called reduced error pruning. More sophisticated pruning methods can be used such as cost complexity pruning where a learning parameter (alpha) is used to weigh whether nodes can be removed based on the size of the sub-tree. This is also known as weakest link pruning.

# Advantages of CART
Simple to understand, interpret, visualize.
Decision trees implicitly perform variable screening or feature selection.
Can handle both numerical and categorical data. Can also handle multi-output problems.
Decision trees require relatively little effort from users for data preparation.
Nonlinear relationships between parameters do not affect tree performance.

# Disadvantages of CART
Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting.
Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This is called variance, which needs to be lowered by methods like bagging and boosting.
Greedy algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees, where the features and samples are randomly sampled with replacement.
Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the data set prior to fitting with the decision tree.
This is all the basic, to get you at par with decision tree learning. An improvement over decision tree learning is made using technique of boosting. A popular library for implementing these algorithms is Scikit-Learn. It has a wonderful api that can get your model up an running with just a few lines of code in python.

Node splitting, or simply splitting, is the process of dividing a node into multiple sub-nodes to create relatively pure nodes. There are multiple ways of doing this, which can be broadly divided into two categories based on the type of target variable:

# Continuous Target Variable
Reduction in Variance

# Categorical Target Variable
Gini Impurity
Information Gain
Chi-Square

Decision Tree Splitting Method #1: Reduction in Variance
Reduction in Variance is a method for splitting the node used when the target variable is continuous, i.e., regression problems. It is so-called because it uses variance as a measure for deciding the feature on which node is split into child nodes.

variance reduction in variance

Variance is used for calculating the homogeneity of a node. If a node is entirely homogeneous, then the variance is zero.

Here are the steps to split a decision tree using reduction in variance:

For each split, individually calculate the variance of each child node
Calculate the variance of each split as the weighted average variance of child nodes
Select the split with the lowest variance
Perform steps 1-3 until completely homogeneous nodes are achieved
The below video excellently explains the reduction in variance using an example:



 

Decision Tree Splitting Method #2: Information Gain
Now, what if we have a categorical target variable? Reduction in variation won’t quite cut it.

Well, the answer to that is Information Gain. Information Gain is used for splitting the nodes when the target variable is categorical. It works on the concept of the entropy and is given by:

information gain

Entropy is used for calculating the purity of a node. Lower the value of entropy, higher is the purity of the node. The entropy of a homogeneous node is zero. Since we subtract entropy from 1, the Information Gain is higher for the purer nodes with a maximum value of 1. Now, let’s take a look at the formula for calculating the entropy:

entropy information gain

Steps to split a decision tree using Information Gain:

For each split, individually calculate the entropy of each child node
Calculate the entropy of each split as the weighted average entropy of child nodes
Select the split with the lowest entropy or highest information gain
Until you achieve homogeneous nodes, repeat steps 1-3
Here’s a video on how to use information gain for splitting a decision tree:



 

Decision Tree Splitting Method #3: Gini Impurity
Gini Impurity is a method for splitting the nodes when the target variable is categorical. It is the most popular and the easiest way to split a decision tree. The Gini Impurity value is:

gini impurity

Wait – what is Gini?

Gini is the probability of correctly labeling a randomly chosen element if it was randomly labeled according to the distribution of labels in the node. The formula for Gini is:

gini decision tree

And Gini Impurity is:

gini impurity decision tree

Lower the Gini Impurity, higher is the homogeneity of the node. The Gini Impurity of a pure node is zero. Now, you might be thinking we already know about Information Gain then, why do we need Gini Impurity?

Gini Impurity is preferred to Information Gain because it does not contain logarithms which are computationally intensive.

Here are the steps to split a decision tree using Gini Impurity:

Similar to what we did in information gain. For each split, individually calculate the Gini Impurity of each child node
Calculate the Gini Impurity of each split as the weighted average Gini Impurity of child nodes
Select the split with the lowest value of Gini Impurity
Until you achieve homogeneous nodes, repeat steps 1-3
And here’s Gini Impurity in video form:



 

Decision Tree Splitting Method #4: Chi-Square
Chi-square is another method of splitting nodes in a decision tree for datasets having categorical target values. It can make two or more than two splits. It works on the statistical significance of differences between the parent node and child nodes.

Chi-Square value is:

chi square decision tree

Here, the Expected is the expected value for a class in a child node based on the distribution of classes in the parent node, and Actual is the actual value for a class in a child node.

The above formula gives us the value of Chi-Square for a class. Take the sum of Chi-Square values for all the classes in a node to calculate the Chi-Square for that node. Higher the value, higher will be the differences between parent and child nodes, i.e., higher will be the homogeneity.

Here are the steps to split a decision tree using Chi-Square:

For each split, individually calculate the Chi-Square value of each child node by taking the sum of Chi-Square values for each class in a node
Calculate the Chi-Square value of each split as the sum of Chi-Square values for all the child nodes
Select the split with higher Chi-Square value
Until you achieve homogeneous nodes, repeat steps 1-3
Of course, there’s a video explaining Chi-Square in the context of a decision tree:



 
