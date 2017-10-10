# cs632
Deep Learning Assignments

# Assignment 1


### Part 1 a:

Main file : part1.py

Classifier file : part1_classifier.py

---

### Part 1 b:
##### 1. In a Nearest Neighbor classifier, is it important that all features be on the same scale?Think: what would happen if one feature ranges between 0-1, and another rangesbetween 0-1000? If it is important that they are on the same scale, how could you achieve this?
In Nearest Neighbor classifier, it is very important to have all features on same scale. If some features range between 0-1 and another ranges between 0-1000, the Euclidean distance formula results can be biased as the 0-1 range is too small as compared to 0-1000.

You can normalize the data on different scales to use it for nearest neighbor. You can use min-max scaling and standardization to get all the features have same scale

##### 2. What is the difference between a numeric and categorical feature? How might you represent a categorical feature so your Nearest Neighbor classifier could work with it?

A categorical feature contains string variables rather than numbers. Example ‘color’ is a categorical feature that can be red, green, blue etc. Numeric value only contains numbers. Example: ‘size of a petal’ can have values like 1 cm, 3 cm etc. 

Encoding, especially One hot encoding can be used to represent categorical features in numeric form, so that Nearest Neighbor classifier can work with it.
Example: Below table shows conversion of categorical feature ‘Device’ with three values ‘Tablet’, ‘Smartphone’, ‘Notebook’ can be represented by D1, D2, D3 respectively.

| Device       | D1  | D2 | D3 |
| :----------- |:---:|---:|---:|
| Smartphone   | 0   |1   | 0  |
| Tablet       | 1   | 0  |0   |
|Notebook      | 0   |0   |1   |

##### 3. What is the importance of testing data?

Test data is used to test the accuracy of the model. Test data helps to evaluate how well a model will perform on instances that it has never seen before.

##### 4. What does “supervised” refer to in “supervised classification”?

Supervised means controlled. Supervised systems learn from labelled data provided by humans. In supervised classification the training data is labeled and algorithms use these labels to classify new data.

##### 5. If you were to include additional features for the Iris dataset, what would they be, and why?
I would consider below additional features:
* Color: Iris setosa, Iris versicolor, Iris virginica flowers have different colors
* Leave’s shape and size: Iris flower plant leaves of each species are quite different from each other
* Seed size: The seed size varies in all three species

