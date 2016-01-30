# Machine Learning Course Project
January 27, 2016  

##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

##Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

For ease of processing files and simplification of code, download the csv files and save to the set working directory.


```r
setwd("~/Desktop/coursera/MachineLearning")
```

Load the libraries we'll be using:

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
library(plotmo)
```

```
## Loading required package: plotrix
```

```
## Warning: package 'plotrix' was built under R version 3.2.3
```

```
## Loading required package: TeachingDemos
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(AppliedPredictiveModeling)
```

With the following code, we'll read the data, and take a quick look at the properties. I also took a quick look at the .csv file by opening the test version in Excel.


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

In the training set, there are 19622 records with 159 variables (the first column is just a numeric count of observations). The "classe" variable we are solving for is divided among 5 classes. 

The test set that was provided contains the exercise readings for 20 participants without the "classe" variable provided. We'll attempt to determine this "classe" with the use of predictive modelling, built using the training set. We'll set the test set to the side until the models are completed. At the end,  we'll apply the same cleaning and transformations to that data, then apply our model. 

###Cleaning the data

We'll clean out the <a href=http://www.inside-r.org/packages/cran/caret/docs/nearZeroVar>NearZeroVariance</a> variables and remove them and the first column (a count of the observations) from our data as these will not contribute to the predictive model.

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
modelFitRF1 <- randomForest(classe ~ ., data=train)
predictionRF1 <- predict(modelFitRF1, test, type = "class")
modelRF <- confusionMatrix(predictionRF1, test$classe)
modelRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    0 1518    6    0    0
##          C    0    0 1362    8    0
##          D    0    0    0 1275    1
##          E    0    0    0    3 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9977          
##                  95% CI : (0.9964, 0.9986)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9971          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9956   0.9914   0.9993
## Specificity            1.0000   0.9991   0.9988   0.9998   0.9995
## Pos Pred Value         1.0000   0.9961   0.9942   0.9992   0.9979
## Neg Pred Value         1.0000   1.0000   0.9991   0.9983   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.1736   0.1625   0.1837
## Detection Prevalence   0.2845   0.1942   0.1746   0.1626   0.1840
## Balanced Accuracy      1.0000   0.9995   0.9972   0.9956   0.9994
```

```r
plot(modelFitRF1)
```

![](MLFinalProject_files/figure-html/randomForest-1.png) 

```r
importance(modelFitRF1)
```

```
##                      MeanDecreaseGini
## user_name                    93.07753
## raw_timestamp_part_1        954.59192
## raw_timestamp_part_2         10.22792
## cvtd_timestamp             1421.31598
## num_window                  585.96803
## roll_belt                   540.24435
## pitch_belt                  286.18407
## yaw_belt                    331.46353
## total_accel_belt            116.57768
## gyros_belt_x                 34.49178
## gyros_belt_y                 51.33270
## gyros_belt_z                110.74807
## accel_belt_x                 62.02930
## accel_belt_y                 71.92824
## accel_belt_z                199.19421
## magnet_belt_x               109.14965
## magnet_belt_y               196.01031
## magnet_belt_z               181.12672
## roll_arm                    112.26850
## pitch_arm                    53.45343
## yaw_arm                      80.82112
## total_accel_arm              26.33487
## gyros_arm_x                  42.66921
## gyros_arm_y                  39.11163
## gyros_arm_z                  18.48576
## accel_arm_x                  89.30064
## accel_arm_y                  52.90548
## accel_arm_z                  37.35796
## magnet_arm_x                101.04174
## magnet_arm_y                 70.08783
## magnet_arm_z                 54.69132
## roll_dumbbell               196.53420
## pitch_dumbbell               83.78152
## yaw_dumbbell                117.92525
## total_accel_dumbbell        120.34213
## gyros_dumbbell_x             39.46029
## gyros_dumbbell_y             84.89189
## gyros_dumbbell_z             22.69485
## accel_dumbbell_x            120.22933
## accel_dumbbell_y            179.91489
## accel_dumbbell_z            132.56321
## magnet_dumbbell_x           238.16563
## magnet_dumbbell_y           327.39478
## magnet_dumbbell_z           295.91850
## roll_forearm                216.14714
## pitch_forearm               308.96655
## yaw_forearm                  55.34029
## total_accel_forearm          33.86167
## gyros_forearm_x              24.70904
## gyros_forearm_y              39.79707
## gyros_forearm_z              25.97830
## accel_forearm_x             130.99499
## accel_forearm_y              42.95575
## accel_forearm_z              90.79754
## magnet_forearm_x             83.07639
## magnet_forearm_y             70.71036
## magnet_forearm_z             92.88319
```

```r
print(modelRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    0 1518    6    0    0
##          C    0    0 1362    8    0
##          D    0    0    0 1275    1
##          E    0    0    0    3 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9977          
##                  95% CI : (0.9964, 0.9986)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9971          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9956   0.9914   0.9993
## Specificity            1.0000   0.9991   0.9988   0.9998   0.9995
## Pos Pred Value         1.0000   0.9961   0.9942   0.9992   0.9979
## Neg Pred Value         1.0000   1.0000   0.9991   0.9983   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.1736   0.1625   0.1837
## Detection Prevalence   0.2845   0.1942   0.1746   0.1626   0.1840
## Balanced Accuracy      1.0000   0.9995   0.9972   0.9956   0.9994
```


###Prediction with Decision Trees
<a href=http://blog.revolutionanalytics.com/2013/06/plotting-classification-and-regression-trees-with-plotrpart.html>Create decision tree</a>

Instead of plotting a decision tree, we can quickly look at a graph of the cross-validation results, and review the confusion matrix resutls and see that the error rate is higher than the random forest method. 


```r
set.seed(4726)
modelFitDT1 <- rpart(classe ~.,method="class", data=train)
predictionDT1 <- predict(modelFitDT1, test, type = "class")
modelDT <- confusionMatrix(predictionDT1,test$classe)
plotcp(modelFitDT1)
```

![](MLFinalProject_files/figure-html/decisionTree-1.png) 

```r
modelDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2151   53    7    4    0
##          B   76 1379  112   10    0
##          C    5   79 1224  130   58
##          D    0    7   15  942   81
##          E    0    0   10  200 1303
## 
## Overall Statistics
##                                          
##                Accuracy : 0.892          
##                  95% CI : (0.885, 0.8988)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.8634         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9637   0.9084   0.8947   0.7325   0.9036
## Specificity            0.9886   0.9687   0.9580   0.9843   0.9672
## Pos Pred Value         0.9711   0.8744   0.8182   0.9014   0.8612
## Neg Pred Value         0.9856   0.9778   0.9773   0.9494   0.9781
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2742   0.1758   0.1560   0.1201   0.1661
## Detection Prevalence   0.2823   0.2010   0.1907   0.1332   0.1928
## Balanced Accuracy      0.9762   0.9386   0.9264   0.8584   0.9354
```

A quick look at the results on the Decision Tree method and we see a lower accuracy rate, 88.73, than the Random Forest accuracy rate of 99.83% witha .17% for our out-of-sample error rate, so we'll progress with the Random Forest for our prediction set.

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
predictionFinal <-predict(modelFitRF1, pmlTest, type="class")
```

And our final results for our 20 test cases.

```r
predictionFinal
```

```
##  2  3 41  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


