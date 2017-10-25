source('RFunctions-1.R')
library(dplyr)
library(e1071)
library(caret)
library(reshape2)
library(ggplot2)

cancer <- read.csv("cancer.csv")
names(cancer) <- c(seq(1,30),"output")
cancer$output <- as.factor(cancer$output)

####################################
#2 Plain SVM
train_idx <- trainTestSplit(cancer,trainPercent=75,seed=5)
train <- cancer[train_idx, ]
test <- cancer[-train_idx, ]

svmfit=svm(output~., data=train, kernel="linear",scale=FALSE)
ypred=predict(svmfit,test)
confusionMatrix(ypred,test$output)

#####################
# Radial SVM unnormalized
train_idx <- trainTestSplit(cancer,trainPercent=75,seed=5)
train <- cancer[train_idx, ]
test <- cancer[-train_idx, ]

svmfit=svm(output~., data=train, kernel="radial",cost=10,scale=FALSE)
ypred=predict(svmfit,test)
confusionMatrix(ypred,test$output)

trainingAccuracy <- NULL
testAccuracy <- NULL
C1 <- c(.01,.1, 1, 10, 20)
for(i in  C1){
    svmfit=svm(output~., data=train, kernel="radial",cost=i,scale=TRUE)
    ypredTrain <-predict(svmfit,train)
    ypredTest=predict(svmfit,test)
    a <-confusionMatrix(ypredTrain,train$output)
    b <-confusionMatrix(ypredTest,test$output)
    trainingAccuracy <-c(trainingAccuracy,a$overall[1])
    testAccuracy <-c(testAccuracy,b$overall[1])
    
}
print(trainingAccuracy)
print(testAccuracy)
a <-rbind(C1,as.numeric(trainingAccuracy),as.numeric(testAccuracy))
b <- data.frame(t(a))
names(b) <- c("C1","trainingAccuracy","testAccuracy")
df <- melt(b,id="C1")
ggplot(df) + geom_line(aes(x=C1, y=value, colour=variable),size=2) +
    xlab("C (SVC regularization)value") + ylab("Accuracy") +
    ggtitle("Training and test accuracy vs C(regularization)")


##################################################################################
# The R equivalent of np.logspace
seqLogSpace <- function(start,stop,len){
  a=seq(log10(10^start),log10(10^stop),length=len)
  10^a
}


library(dplyr)
cancer <- read.csv("cancer.csv")
names(cancer) <- c(seq(1,30),"output")
cancer$output <- as.factor(cancer$output)


set.seed(6)


acc
param_range = seqLogSpace(-3,2,20)
# Loop through values of C
accuracy <- NULL
for(i in param_range){
    # Set no of folds
    noFolds=5
    # Create the rows which fall into different folds from 1..noFolds
    folds = sample(1:noFolds, nrow(cancer), replace=TRUE) 
    foldAccuracy<-0
    # Loop through the folds
    for(j in 1:noFolds){
        # The training is all rows for which the row is != j (k-1 folds -> training)
        train <- cancer[folds!=j,]
        # The rows which have j as the index become the test set
        test <- cancer[folds==j,]
        # Create a SVM model for this
        svmfit=svm(output~., data=train, kernel="radial",cost=i,scale=TRUE)

        ypred=predict(svmfit,test)
        a <-confusionMatrix(ypred,test$output)
        testAccuracy <-a$overall[1]
        # Add all the Cross Validation errors
        foldAccuracy=foldAccuracy+testAccuracy
    }
    # Compute the average of MSE for K folds for number of features 'i'
    accuracy=c(accuracy,foldAccuracy/noFolds)
    print(accuracy)
}

a <- seq(1,13)
d <- as.data.frame(t(rbind(a,cvError)))
names(d) <- c("Features","CVError")
#Plot the CV Error vs No of Features
ggplot(d,aes(x=Features,y=CVError),color="blue") + geom_point() + geom_line(color="blue") +
    xlab("No of features") + ylab("Cross Validation Error") +
    ggtitle("Forward Selection - Cross Valdation Error vs No of Features")


# The R equivalent of np.logspace
seqLogSpace <- function(start,stop,len){
    a=seq(log10(10^start),log10(10^stop),length=len)
    10^a
}
