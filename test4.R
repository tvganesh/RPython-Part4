
# Part 4 - Test
source('RFunctions-1.R')
library(dplyr)
library(e1071)
library(caret)
library(reshape2)
library(ggplot2)
library(rpart)
library(rpart.plot)

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
cancer <- read.csv("cancer.csv")
names(cancer) <- c(seq(1,30),"output")
cancer$output <- as.factor(cancer$output)

set.seed(6)


# Create the range of C1 in log space
param_range = seqLogSpace(-3,2,20)
# Initialize the overall training and test accuracy to NULL
overallTrainAccuracy <- NULL
overallTestAccuracy <- NULL

# Loop over the parameter range of Gamma
for(i in param_range){
    # Set no of folds
    noFolds=5
    # Create the rows which fall into different folds from 1..noFolds
    folds = sample(1:noFolds, nrow(cancer), replace=TRUE) 
    # Initialize the training and test accuracy of folds to 0
    trainingAccuracy <- 0
    testAccuracy <- 0
    
    # Loop through the folds
    for(j in 1:noFolds){
        # The training is all rows for which the row is != j (k-1 folds -> training)
        train <- cancer[folds!=j,]
        # The rows which have j as the index become the test set
        test <- cancer[folds==j,]
        # Create a SVM model for this
        svmfit=svm(output~., data=train, kernel="radial",gamma=i,scale=TRUE)
  
        # Add up all the fold accuracy for training and test separately  
        ypredTrain <-predict(svmfit,train)
        ypredTest=predict(svmfit,test)
        
        # Create confusion matrix 
        a <-confusionMatrix(ypredTrain,train$output)
        b <-confusionMatrix(ypredTest,test$output)
        # Get the accuracy
        trainingAccuracy <-trainingAccuracy + a$overall[1]
        testAccuracy <-testAccuracy+b$overall[1]

    }
    # Compute the average of accuracy for K folds for number of features 'i'
    overallTrainAccuracy=c(overallTrainAccuracy,trainingAccuracy/noFolds)
    overallTestAccuracy=c(overallTestAccuracy,testAccuracy/noFolds)
}
#Create a dataframe
a <- rbind(param_range,as.numeric(overallTrainAccuracy),
               as.numeric(overallTestAccuracy))
b <- data.frame(t(a))
names(b) <- c("C1","trainingAccuracy","testAccuracy")
df <- melt(b,id="C1")
#Plot in log axis
ggplot(df) + geom_line(aes(x=C1, y=value, colour=variable),size=2) +
      xlab("C (SVC regularization)value") + ylab("Accuracy") +
      ggtitle("Training and test accuracy vs C(regularization)") + scale_x_log10()
#}



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



rpart = NULL
m <-rpart(Species~.,data=iris)
rpart.plot(m,extra=2,main="Decision Tree - IRIS")


#https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the dataset
cancer <- read.csv("cancer.csv")
names(cancer) <-c('mean radius', 'mean texture', 'mean perimeter', 'mean area',
                  'mean smoothness', 'mean compactness', 'mean concavity',
                  'mean concave points', 'mean symmetry', 'mean fractal dimension',
                  'radius error', 'texture error', 'perimeter error', 'area error',
                  'smoothness error', 'compactness error', 'concavity error',
                  'concave points error', 'symmetry error', 'fractal dimension error',
                  'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                  'worst smoothness', 'worst compactness', 'worst concavity',
                  'worst concave points', 'worst symmetry', 'worst fractal dimension','target')
#names(cancer) <- c(seq(1,30),"output")
cancer$target <- as.factor(cancer$target)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(target~., data=cancer, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


########################
#Dummy classifier
source('RFunctions-1.R')
library(dplyr)
library(e1071)
library(caret)
library(reshape2)
library(ggplot2)

cancer <- read.csv("cancer.csv")
names(cancer) <- c(seq(1,30),"target")
cancer$target <- as.factor(cancer$target)

####################################
#2 Plain SVM
train_idx <- trainTestSplit(cancer,trainPercent=75,seed=5)
train <- cancer[train_idx, ]
test <- cancer[-train_idx, ]

#Dummy classifier majoritrity class
count <- sum(train$output==1)/dim(train)[1]
RsquaredDummyClassifier <- function(train,test,type="majority"){
  if(type=="majority"){
      count <- sum(train$target==1)/dim(train)[1]
  }
  count
}

#################################
#PRROC
source("RFunctions-1.R")
library(dplyr)
library(caret)
library(e1071)
library(PRROC)
# Read the data (from sklearn)
d <- read.csv("digits.csv")
digits <- d[2:66]
digits$X64 <- as.factor(digits$X64)


# Split as training and test sets
train_idx <- trainTestSplit(digits,trainPercent=75,seed=5)
train <- digits[train_idx, ]
test <- digits[-train_idx, ]

svmfit=svm(X64~., data=train, kernel="linear",scale=FALSE,probability=TRUE)

ypred=predict(svmfit,test,probability=TRUE)

a <- test$X64==1
b <- test$X64==0
m0<-attr(ypred,"probabilities")[a,1]
m1<-attr(ypred,"probabilities")[b,1]

scores <- data.frame(m0,test$X64)

pr <- pr.curve(scores.class0=scores[scores$test.X64=="0",]$m0,
               scores.class1=scores[scores$test.X64=="1",]$m0,
               curve=T)

plot(pr)

############   Works
source("RFunctions-1.R")
library(dplyr)
library(caret)
library(e1071)
library(PRROC)
# Read the data (from sklearn)
d <- read.csv("digits.csv")
digits <- d[2:66]
digits$X64 <- as.factor(digits$X64)


# Split as training and test sets
train_idx <- trainTestSplit(digits,trainPercent=75,seed=5)
train <- digits[train_idx, ]
test <- digits[-train_idx, ]

svmfit=svm(X64~., data=train, kernel="linear",scale=FALSE,probability=TRUE)
ypred=predict(svmfit,test,probability=TRUE)
head(attr(ypred,"probabilities"))

m0<-attr(ypred,"probabilities")[,1]
m1<-attr(ypred,"probabilities")[,2]

scores <- data.frame(m1,test$X64)
pr <- pr.curve(scores.class0=scores[scores$test.X64=="1",]$m1,
               scores.class1=scores[scores$test.X64=="0",]$m1,
               curve=T)

plot(pr)




####

pr<-pr.curve(m0, m1,curve=TRUE)
plot(pr)

oc<-roc.curve(m0, m1,curve=TRUE)
plot(roc)


ypred=predict(svmfit,test,decision.values=TRUE)

a <-attr(ypred,"decision.values")


scores <- data.frame(a,test$X64)
pr <- pr.curve(scores.class0=scores[scores$test.X64=="1",]$X0.1,
               scores.class1=scores[scores$test.X64=="0",]$X0.1,
               curve=T)

m0<-attr(ypred,"probabilities")[,1]
m1<-attr(ypred,"probabilities")[,2]

pr<-pr.curve(m0, m1,curve=TRUE)
plot(pr)

roc<-roc.curve(m0, m1,curve=TRUE)
plot(roc)

svmfit=svm(X64~., data=train, kernel="radial",scale=TRUE)
ypred=predict(svmfit,test)
confusionMatrix(ypred,test$X64)

# Fit a generalized linear logistic model, 
fit=glm(X64~.,family=binomial,data=train,control = list(maxit = 100))
#fit=glm(X64~.,family=binomial,data=train)
# Predict the output from the model
a=predict(fit,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b=ifelse(a>0.5,1,0)
# Compute the confusion matrix for training data
confusionMatrix(b,train$output)

m=predict(fit,newdata=test,type="response")
x <- m[m>0.5]
y <- m[m<=0.5]
pr <- pr.curve( x, y, curve = TRUE )
plot(pr)
n=ifelse(m>0.5,1,0)
# Compute the confusion matrix for test output
confusionMatrix(n,test$output)


######################################################################
source("RFunctions-1.R")
library(dplyr)
library(precrec)
# Read the data (from sklearn)
d <- read.csv("digits.csv")
digits <- d[2:66]
digits$X64 <- as.factor(digits$X64)


# Split as training and test sets
train_idx <- trainTestSplit(digits,trainPercent=75,seed=5)
train <- digits[train_idx, ]
test <- digits[-train_idx, ]

svmfit=svm(X64~., data=train, kernel="linear",scale=FALSE)

#ypred=predict(svmfit,test)
ypred=predict(svmfit,test,decision.values=TRUE)

a <-attr(ypred,"decision.values")

ab <- test$X64==1
c <- test$X64==0
m0 <-a[b]
m1 <-a[c]
pr<-pr.curve(m1, m0,curve=TRUE)
plot(pr)

roc<-roc.curve(m1, m0,curve=TRUE)
plot(roc)

attr(,"decision.values")


a <-as.numeric(as.character(ypred))
b <- as.numeric(as.character(test$X64))

pred <- prediction(a,b)
perf <- performance(pred, "prec", "rec")

# Recall-Precision curve             


plot (perf,colorize=T)

