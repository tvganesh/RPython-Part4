source('RFunctions-1.R')
library(dplyr)
library(e1071)
library(caret)
library(reshape2)
library(ggplot2)

cancer <- read.csv("cancer.csv")
names(cancer) <- c(seq(1,30),"output")


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
