library(data.table)   # Fast I/O
library(plyr)
library(dplyr)        # Data munging
library(tidyr)        # Data munging
library(lubridate)    # Makes dates easy
library(gplots)
library(ggplot2)
library(plotly)       # Interactive charts
library(magrittr)     # pipe operators
library(lattice)
library(caret)        # Handy ML functions
library(rpart)        # Decision Trees
library(rpart.plot)   # Pretty tree plots
library(ROCR)         # ML evaluation
library(e1071)        # Misc stat fns
library(randomForest) # rf
library(caTools)
library(glm.predict)
library(gridExtra)
library(ggthemes)
library(MASS)

> str(churn)
> sapply(churn, function(x) sum(is.na(x)))
> churn <- churn[complete.cases(churn), ]
> print(str(churn))

> churn$MultipleLines[churn$MultipleLines=="No phone service"] <- "No"
> churn$OnlineSecurity[churn$OnlineSecurity=="No internet service"] <- "No"
> churn$OnlineBackup[churn$OnlineBackup=="No internet service"] <- "No"
> churn$DeviceProtection[churn$DeviceProtection=="No internet service"] <- "No"
> churn$TechSupport[churn$TechSupport=="No internet service"] <- "No"
> churn$StreamingTV[churn$StreamingTV=="No internet service"] <- "No"
> churn$StreamingMovies[churn$StreamingMovies=="No internet service"] <- "No"

> min(churn$tenure); max(churn$tenure)
> group_tenure <- function(tenure){
  +     if (tenure >= 0 && tenure <= 12){
    +         return('0-12 month')
    +     }else if (tenure > 12 && tenure <= 24){
      +         return('12-24 month')
      +     }else if (tenure > 24 && tenure <= 36){
        +         return('24-36 month')
        +     }else if (tenure > 36 && tenure <= 48){
          +         return('36-48 month')
          +     }else if (tenure > 48 && tenure <= 60){
            +         return('48-60 month')
            +     }else if (tenure > 60){
              +         return('>60 month')
              +     }
  + }
> churn$tenure_interval <- sapply(churn$tenure,group_tenure)
> churn$tenure_interval <- as.factor(churn$tenure_interval)
> head(churn,n=3)# A tibble: 3 x 22

> churn$SeniorCitizen[churn$SeniorCitizen==1] <- "Yes"
> churn$SeniorCitizen[churn$SeniorCitizen==0] <- "No"

> churn$customerID <- NULL
> churn$tenure <- NULL

> library(corrplot)
> numeric.var <- sapply(churn, is.numeric)
> corr.matrix <- cor(churn[,numeric.var])
> corrplot(corr.matrix, main="\n\nCorrelation Plot for Numerical Variables", method="number")

> ggplot(churn, aes(x=Partner)) + ggtitle("Partner") + xlab("Partner") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
> ggplot(churn, aes(x=Dependents)) + ggtitle("Dependents") + xlab("Dependents") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
> ggplot(churn, aes(x=PhoneService)) + ggtitle("Phone Service") + xlab("Phone Service") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
> ggplot(churn, aes(x=MultipleLines)) + ggtitle("Multiple Lines") + xlab("Multiple Lines") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
> ggplot(churn, aes(x=InternetService)) + ggtitle("Internet Service") + xlab("Internet Service") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
> ggplot(churn, aes(x=OnlineSecurity)) + ggtitle("Online Security") + xlab("Online Security") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
> ggplot(churn, aes(x=OnlineBackup)) + ggtitle("Online Backup") + xlab("Online Backup") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
> ggplot(churn, aes(x=DeviceProtection)) + ggtitle("Device Protection") + xlab("Device Protection") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
> ggplot(churn, aes(x=TechSupport)) + ggtitle("Tech Support") + xlab("Tech Support") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
> ggplot(churn, aes(x=StreamingTV)) + ggtitle("Streaming TV") + xlab("Streaming TV") + geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()

> churn$Churn <- as.factor(churn$Churn)
> sample <- sample.split(churn$Churn,SplitRatio = 7/10)
> trainData <- subset(churn,sample==TRUE)
> TestData <- subset(churn,sample==FALSE)
> dim(trainData); dim(TestData)

# Logistic Regression Model

> Logistic.model <- glm(Churn ~ .,family=binomial(link="logit"),data=trainData)
> summary(Logistic.model)


# Feature Analysis: The top three most-relevant features include Contract, tenure_interval and PaperlessBilling.
> anova(Logistic.model, test="Chisq")

# Assessing the predictive ability of the Logistic Regression model
> TestData$Churn <- as.character(TestData$Churn)
> TestData$Churn[TestData$Churn=="No"] <- "0"
> TestData$Churn[TestData$Churn=="Yes"] <- "1"
> fitted.results <- predict(Logistic.model,newdata=TestData,type='response')
> fitted.results <- ifelse(fitted.results > 0.5,1,0)
> misClasificError <- mean(fitted.results != TestData$Churn)
> print(paste('Logistic Regression Accuracy',1-misClasificError))
[1] "Logistic Regression Accuracy 0.796682464454976"

# Logistic Regression Confusion Matrix
> print("Confusion Matrix for Logistic Regression"); table(TestData$Churn, fitted.results > 0.5)

# Odds Ratio
> library(MASS)
> exp(cbind(OR=coef(Logistic.model), confint(Logistic.model)))


# Decision Tree Model
# use "Contract", "tenure_interval" and "PaperlessBilling" for plotting Decision Trees.
> library(grid)
library(libcoin)
library(mvtnorm)
library(partykit)

> summary(churn)
> fit <- rpart(Churn ~ Contract + tenure_interval + PaperlessBilling,
+              method="class",
+              data=trainData)
> summary(fit)
Call:
rpart(formula = Churn ~ Contract + tenure_interval + PaperlessBilling, 
data = trainData, method = "class")
n= 4922 

CP nsplit rel error    xerror       xstd
1 0.04485219      0 1.0000000 1.0000000 0.02369296
2 0.01000000      3 0.8654434 0.8654434 0.02257170

Variable importance
Contract  tenure_interval PaperlessBilling 
53               36               11 

Node number 1: 4922 observations,    complexity param=0.04485219
predicted class=No   expected loss=0.2657456  P(node) =1
class counts:  3614  1308
probabilities: 0.734 0.266 
left son=2 (2224 obs) right son=3 (2698 obs)
Primary splits:
Contract         splits as  RLL,    improve=313.32430, (0 missing)
tenure_interval  splits as  LRLLLL, improve=191.73500, (0 missing)
PaperlessBilling splits as  LR,     improve= 76.76458, (0 missing)
Surrogate splits:
tenure_interval  splits as  LRRRLL, agree=0.792, adj=0.540, (0 split)
PaperlessBilling splits as  LR,     agree=0.581, adj=0.072, (0 split)

Node number 2: 2224 observations
predicted class=No   expected loss=0.0692446  P(node) =0.4518488
class counts:  2070   154
probabilities: 0.931 0.069 

Node number 3: 2698 observations,    complexity param=0.04485219
predicted class=No   expected loss=0.4277242  P(node) =0.5481512
class counts:  1544  1154
probabilities: 0.572 0.428 
left son=6 (938 obs) right son=7 (1760 obs)
Primary splits:
PaperlessBilling splits as  LR,     improve=43.38121, (0 missing)
tenure_interval  splits as  LRLLLL, improve=43.24976, (0 missing)

Node number 6: 938 observations
predicted class=No   expected loss=0.3049041  P(node) =0.1905729
class counts:   652   286
probabilities: 0.695 0.305 

Node number 7: 1760 observations,    complexity param=0.04485219
predicted class=No   expected loss=0.4931818  P(node) =0.3575782
class counts:   892   868
probabilities: 0.507 0.493 
left son=14 (928 obs) right son=15 (832 obs)
Primary splits:
tenure_interval splits as  LRLLLL, improve=40.00347, (0 missing)

Node number 14: 928 observations
predicted class=No   expected loss=0.3922414  P(node) =0.1885412
class counts:   564   364
probabilities: 0.608 0.392 

Node number 15: 832 observations
predicted class=Yes  expected loss=0.3942308  P(node) =0.169037
class counts:   328   504
probabilities: 0.394 0.606 
> rpart.plot(fit,type=4,extra=4)
> rpart.plot(fit,type=4,extra=2)
> predict(fit,TestData,type="prob")



# tree model method2

> trainData$Churn <- as.factor(trainData$Churn)
> trainData$Contract <- as.factor(trainData$Contract)
> trainData$tenure_interval <- as.factor(trainData$tenure_interval)
> trainData$PaperlessBilling <- as.factor(trainData$PaperlessBilling)
> tree <- ctree(Churn ~ Contract + tenure_interval + PaperlessBilling, trainData)
> plot(tree)

> trainData$SeniorCitizen <- as.factor(trainData$SeniorCitizen)
> trainData$SeniorCitizen <- as.factor(trainData$SeniorCitizen)
> trainData$MultipleLines <- as.factor(trainData$MultipleLines)
> trainData$PaymentMethod <- as.factor(trainData$PaymentMethod)
> tree1 <- ctree(Churn ~ Contract + tenure_interval + PaperlessBilling + SeniorCitizen + MultipleLines + PaymentMethod, trainData)
> plot(tree1)

> TestData$Churn <- as.factor(TestData$Churn)
> TestData$Contract <- as.factor(TestData$Contract)
> TestData$tenure_interval <- as.factor(TestData$tenure_interval)
> TestData$PaperlessBilling <- as.factor(TestData$PaperlessBilling)
> TestData$SeniorCitizen <- as.factor(TestData$SeniorCitizen)
> TestData$MultipleLines <- as.factor(TestData$MultipleLines)
> TestData$PaymentMethod <- as.factor(TestData$PaymentMethod)

# Decision Tree Confusion Matrix
> pred_tree <- predict(tree, TestData)
> print("Confusion Matrix for Decision Tree"); table(Predicted = pred_tree, Actual = TestData$Churn)
[1] "Confusion Matrix for Decision Tree"
Actual
Predicted    0    1
No  1378  332
Yes  171  229
> pred_tree1 <- predict(tree1, TestData)
> print("Confusion Matrix for Decision Tree"); table(Predicted = pred_tree1, Actual = TestData$Churn)
[1] "Confusion Matrix for Decision Tree"
Actual
Predicted    0    1
No  1346  295
Yes  203  266

# Decision Tree Accuracy
> p1 <- predict(tree, trainData)
> tab1 <- table(Predicted = p1, Actual = trainData$Churn)
> tab2 <- table(Predicted = pred_tree, Actual = TestData$Churn)
> print(paste('Decision Tree Accuracy',sum(diag(tab2))/sum(tab2)))
[1] "Decision Tree Accuracy 0.761611374407583"
> p11 <- predict(tree1, trainData)
> tab11 <- table(Predicted = p11, Actual = trainData$Churn)
> tab21 <- table(Predicted = pred_tree1, Actual = TestData$Churn)
> print(paste('Decision Tree Accuracy',sum(diag(tab21))/sum(tab21)))
[1] "Decision Tree Accuracy 0.763981042654028"


# Random Forest Model
> churn$Churn = factor(churn$Churn)
> rfModel <- randomForest(Churn ~., data = trainData)
> print(rfModel)

# Random Forest Prediction and Confusion Matrix
> pred_rf <- predict(rfModel, TestData)
> caret::confusionMatrix(pred_rf, TestData$Churn)
> plot(rfModel)

# Tune Random Forest Model
> t <- tuneRF(trainData[, -18], trainData[, 18], stepFactor = 0.5, plot = TRUE, ntreeTry = 200, trace = TRUE, improve = 0.05)

# Fit the Random Forest Model After Tuning
> rfModel_new <- randomForest(Churn ~., data = trainData, ntree = 200, mtry = 2, importance = TRUE, proximity = TRUE)
> print(rfModel_new)

# Random Forest Predictions and Confusion Matrix After Tuning
> pred_rf_new <- predict(rfModel_new, TestData)
> caret::confusionMatrix(pred_rf_new, TestData$Churn)

# Random Forest Feature Importance
> varImpPlot(rfModel_new, sort=T, n.var = 10, main = 'Top 10 Feature Importance')


