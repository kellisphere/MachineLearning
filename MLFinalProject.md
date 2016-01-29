# Machine Learning Course Project
January 27, 2016  

##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

##Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>
For ease of processing files and simplification of code, download the csv files and save to the set working directory, MachineLearning on the desktop.


```r
setwd("~/Desktop/coursera/MachineLearning")
```


```r
## Load the preferred libraries 
library(caret) 
```

```
## Warning: package 'caret' was built under R version 3.2.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rpart)
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(AppliedPredictiveModeling)
```

Two sets of data were provided, training and test. 

The Test set contains the exercise readings for 20 participants without the "classe" variable of exercise provided. We will attempt to determine this "classe" with the use of predictive modelling, built using the training set. We will set the Test set to the side until the models are completed.
 
 With the following code, we'll read the data.


```r
pmlTrain <- read.csv("pml-training.csv", header=TRUE, na.strings=c("NA","#Div/0!"))  ## The training set
pmlTest <- read.csv("pml-testing.csv", header=TRUE, na.string=c("NA", "#Div/0!")) ## The test set -set aside
dim(pmlTrain)
```

```
## [1] 19622   160
```

```r
dim(pmlTest)
```

```
## [1]  20 160
```

```r
summary(pmlTrain$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

There are 19622 records with 159 variables (the first column is just a numeric count of observations). The "classe" variable we are solving for is divided among 5 classes. 

###Cleaning the data

We'll clean out the <a href=http://www.inside-r.org/packages/cran/caret/docs/nearZeroVar>NearZeroVariance</a> variables and remove them from our data, and remove the first column, a count of the observations.

```r
nzv <- nearZeroVar(pmlTrain, saveMetrics=TRUE)  ## remove nearZeroVariances
pmlTrain <- pmlTrain[,nzv$nzv==FALSE]
pmlTrain <- pmlTrain[c(-1)]  ## remove first column (count)
```

Remove observations with 75% NA:

```r
noNAs<- pmlTrain ## find and remove 75% of NAs
for(i in 1:length(pmlTrain)) {
    if( sum( is.na( pmlTrain[, i] ) ) /nrow(pmlTrain) >= 0.75) {
        for(j in 1:length(noNAs)) {
            if( length( grep(names(pmlTrain[i]), names(noNAs)[j]) ) == 1)  {
                noNAs <- noNAs[ , -j]
            }   
        } 
    }
}
pmlTrain <- noNAs ## set back to name
rm(noNAs) ## remove excess data
dim(pmlTrain)
```

```
## [1] 19622    58
```
This brings us down to 58 columns.

### Split data

Now we'll split the data into a 60/40 training/test set to train the model then test the model before using for our prediction on the 20 observations in the final set. 


```r
inTrain <- createDataPartition(y=pmlTrain$classe,p=.60,list=FALSE)
train <-pmlTrain[inTrain,]
test <- pmlTrain[-inTrain,]
```

```r
dim(train)
```

```
## [1] 11776    58
```

```r
dim(test)
```

```
## [1] 7846   58
```


###Prediction with Random Forests
For <a href=http://www.statmethods.net/advstats/cart.html>Random Forest information</a>

```r
set.seed(4726)
modelFitA1 <- randomForest(classe ~ ., data=train)
predictionA1 <- predict(modelFitA1, test, type = "class")
modelRF <- confusionMatrix(predictionA1, test$classe)
modelRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    1    0    0    0
##          B    1 1517    5    0    0
##          C    0    0 1361    0    0
##          D    0    0    2 1286    2
##          E    0    0    0    0 1440
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9986          
##                  95% CI : (0.9975, 0.9993)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9982          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9993   0.9949   1.0000   0.9986
## Specificity            0.9998   0.9991   1.0000   0.9994   1.0000
## Pos Pred Value         0.9996   0.9961   1.0000   0.9969   1.0000
## Neg Pred Value         0.9998   0.9998   0.9989   1.0000   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1933   0.1735   0.1639   0.1835
## Detection Prevalence   0.2845   0.1941   0.1735   0.1644   0.1835
## Balanced Accuracy      0.9997   0.9992   0.9974   0.9997   0.9993
```

```r
plot(modelFitA1)
```

![](MLFinalProject_files/figure-html/randomForest-1.png) 

```r
importance(modelFitA1)
```

```
##                      MeanDecreaseGini
## user_name                   95.718345
## raw_timestamp_part_1       982.764090
## raw_timestamp_part_2         9.673919
## cvtd_timestamp            1393.992841
## num_window                 588.527582
## roll_belt                  546.827101
## pitch_belt                 299.638480
## yaw_belt                   334.964848
## total_accel_belt            99.559383
## gyros_belt_x                38.779201
## gyros_belt_y                51.468809
## gyros_belt_z               124.432224
## accel_belt_x                59.918307
## accel_belt_y                66.597343
## accel_belt_z               196.824369
## magnet_belt_x              110.866417
## magnet_belt_y              198.553920
## magnet_belt_z              185.528124
## roll_arm                   125.305474
## pitch_arm                   57.322928
## yaw_arm                     81.523625
## total_accel_arm             27.130126
## gyros_arm_x                 41.554543
## gyros_arm_y                 41.647606
## gyros_arm_z                 18.116645
## accel_arm_x                 92.246388
## accel_arm_y                 50.561686
## accel_arm_z                 38.837497
## magnet_arm_x                92.327018
## magnet_arm_y                73.894905
## magnet_arm_z                57.061664
## roll_dumbbell              187.594279
## pitch_dumbbell              83.497067
## yaw_dumbbell               103.647975
## total_accel_dumbbell       115.031643
## gyros_dumbbell_x            39.282592
## gyros_dumbbell_y            96.095199
## gyros_dumbbell_z            22.141805
## accel_dumbbell_x           125.366950
## accel_dumbbell_y           177.252175
## accel_dumbbell_z           138.047520
## magnet_dumbbell_x          245.087768
## magnet_dumbbell_y          321.134301
## magnet_dumbbell_z          290.462288
## roll_forearm               228.083049
## pitch_forearm              289.988371
## yaw_forearm                 52.255099
## total_accel_forearm         33.310660
## gyros_forearm_x             23.486325
## gyros_forearm_y             37.190887
## gyros_forearm_z             26.347075
## accel_forearm_x            130.504249
## accel_forearm_y             43.201209
## accel_forearm_z             87.633855
## magnet_forearm_x            69.013983
## magnet_forearm_y            71.646365
## magnet_forearm_z            90.094905
```

```r
print(modelRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    1    0    0    0
##          B    1 1517    5    0    0
##          C    0    0 1361    0    0
##          D    0    0    2 1286    2
##          E    0    0    0    0 1440
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9986          
##                  95% CI : (0.9975, 0.9993)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9982          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9993   0.9949   1.0000   0.9986
## Specificity            0.9998   0.9991   1.0000   0.9994   1.0000
## Pos Pred Value         0.9996   0.9961   1.0000   0.9969   1.0000
## Neg Pred Value         0.9998   0.9998   0.9989   1.0000   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1933   0.1735   0.1639   0.1835
## Detection Prevalence   0.2845   0.1941   0.1735   0.1644   0.1835
## Balanced Accuracy      0.9997   0.9992   0.9974   0.9997   0.9993
```


###Prediction with Decision Trees
<a href=http://blog.revolutionanalytics.com/2013/06/plotting-classification-and-regression-trees-with-plotrpart.html>Create decision tree</a>

Instead of plotting a decision tree, we can quickly look at a graph of the cross-validation results, and see that the error rate is higher than the random forest method. 


```r
set.seed(4726)
modelFitB1 <- rpart(classe ~., method="class", data=train)
plot(modelFitB1)
text(modelFitB1)
```

![](MLFinalProject_files/figure-html/decisionTree-1.png) 

```r
plotcp(modelFitB1)
```

![](MLFinalProject_files/figure-html/decisionTree-2.png) 

A quick look at the results on teh decision tree and we see the variables we're looking for intermingled. The Random Forest has a 99.83% accuracy rate, a .17% out-of-sample error rate, so we'll progress with this for our prediction set.

###Predicting our results
First, we'll use the same cleaning methods as above:

```r
cleanFormat <- colnames(train[,-58]) # classe column removal
pmlTest <-pmlTest[cleanFormat]
dim(pmlTest)
```

```
## [1] 20 57
```

And we'll coerce the data into the same format:


```r
for (i in 1:length(pmlTest) ) {
    for(j in 1:length(train)) {
        if( length( grep(names(train[i]), names(pmlTest)[j]) ) == 1)  {
            class(pmlTest[j]) <- class(train[i])
        }      
    }      
}

# To get the same class between pmlTest and train
pmlTest <- rbind(train[2,-58], pmlTest) ## remove excess rows
pmlTest <- pmlTest[-1,]
```

Then we apply the prediction model to the data:

```r
predictionFinal <-predict(modelFitA1, pmlTest, type="class")
```

And our final results.

```r
predictionFinal
```

```
##  2  3 41  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


