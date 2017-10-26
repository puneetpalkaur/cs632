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

---

### Part 2 a: 
Main file: part2.py

Classifier file: part1_classifier.py

Assumptions: csv file names start with ‘0’ (zero) except for label. This is to differentiate label and email dataset if these all exist in same folder

### Part 2 b:

##### 1. What are the strengths and weaknesses of using a Bag of Words? (Tip: would this representation let you distinguish between two sentences the same words, but in adifferent order?)


•	Strengths of BOW: 

a)	It is simple and easy to implement

b)	BOW is effective in document classifications

•	Weaknesses of BOW: 

a)	It disregards grammar and word order

b)	It is susceptible to bias and poisoning as many spammers tricks BOW by including additional text in emails to make sure their email is identified as ham instead of spam

c)	BOW can be less useful in cases spammer replace text of an email with an image


##### 2. Which of your features do you think is most predictive, least predictive, and why? 
Most predictive features can be the sender’s email, subject of email and time of the day the email was sent. 

•	Most spam emails have vague email ids and domain names like ‘flordevil_pingam@t-online.de’ or ‘SergioBedpsti@’.

•	Spam email subjects starts with numbers and contain percentages and can have prefixes like “Re:” even though it is not a Reply to any existing email conversation.

•	Spams are usually auto generated emails sent at odd hours like at mid night and early mornings.

Least predictive features can be looking for particular words or symbols. Example: does the document contains the word "drugs",because sometimes a genuine email contains the same word and can be marked as spam.

	

##### 3. Did your classifier misclassify any examples? Why or why not?
Yes, my model misclassified few examples. This is because sometimes an email which is not a spam might contain words which are identified as spam words. Example: use of dollar sign in a genuine email might make it categorized as a spam.

---
# Assignment 2


### Part 1 :

Main file : train.py
Instructions : 

•	This file takes training data path and validation data path through command line arguements 

•	It saves the trained model as 'trained_model_1.h5' at your default path

### Part 2 :

Main file : predict.py
Instructions : 

•	Pass the test file name and model name while running the file

•	Output will be saved to 'predictions.txt' file at your default path
