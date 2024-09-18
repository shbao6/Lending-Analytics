setwd("/Users/shiqi/Downloads/Predictive Modeler")
library(tidyverse)
library(dplyr)
library("readxl")
data <- read_excel("Lending Analtyics PM Assessment 2022.xlsx", sheet = "Dataset")

# charge off risk by random forest
library(randomForest)
require(caTools)
summary(data)
sapply(data, class)
colnames(data)
data <- transform(data, Times_DQ_L12 = as.numeric(Times_DQ_L12),
                  Origination_LTV = as.numeric(Origination_LTV),
                  chgoff_flag = as.factor(chgoff_flag),
                  Direct_Deposit_flag = as.factor(Direct_Deposit_flag),
                  New_Customer_flag = as.factor(New_Customer_flag),
                  Never_DQ_flag = as.factor(Never_DQ_flag),
                  modification_flag = as.factor(modification_flag))
# check for missing values
colSums(is.na(data))
# Times_DQ_L12 NA to 0 times be delinquent
data[["Times_DQ_L12"]][is.na(data[["Times_DQ_L12"]])] <- 0
# impute origination LTV and TotalDeposit_Amount
trimmed_data = data %>% select(Collateral_code,APR,orig_term,loan_amount,
                               Balance,Payment_amount,Days_Past_Due,Payments_remaining,
                               Sched_Payment_Amt,term,Num_Payments_Made,Amount_Past_Due,
                               Times_DQ_L12,Product,TotalDeposit_Amount,Customership_Tenure,DQ_Status,Origination_LTV,
                               Lending_Product_Count,Customer_Cohort,chgoff_flag,current_MOB,Tier,geo_region,
                               Generation,firstTime_borrower,Never_DQ_flag,
                               modification_flag) 
colSums(is.na(trimmed_data))
trimmed_data$Origination_LTV[is.na(trimmed_data$Origination_LTV)] <- median(trimmed_data$Origination_LTV, na.rm = T) 
trimmed_data$TotalDeposit_Amount[is.na(trimmed_data$TotalDeposit_Amount)] <- median(trimmed_data$TotalDeposit_Amount, na.rm = T) 
# character to factor
trimmed_data <- as.data.frame(unclass(trimmed_data), stringsAsFactors = TRUE)
sapply(trimmed_data, class)
library(caret)
sample = sample.split(trimmed_data$chgoff_flag, SplitRatio = .8)
train = subset(trimmed_data, sample == TRUE)
test  = subset(trimmed_data, sample == FALSE)

# use 10-fold cross validation
trControl <- trainControl(method = "cv",number = 10, search = "grid")
set.seed(1234)
# Run the model
rf_default <- train(chgoff_flag~.,data = train, method = "rf", metric = "Accuracy", trControl = trControl)
# Print the results
print(rf_default)
# best mtry = 34
best_mtry = 34
# search the best max nodes - tree depth
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(15: 30)) {
  set.seed(1234)
  rf_maxnode <- train(chgoff_flag~.,
                      data = train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
# best maxnode = 22
fit_rf <- train(chgoff_flag~.,
                train,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                #ntree = 500,
                maxnodes = 22)

prediction <-predict(fit_rf, test)
confusionMatrix(prediction, test$chgoff_flag)
# AUC and ROC plot
prediction <- predict(fit_rf, test,type='prob')
roc_obj <- roc(test$chgoff_flag, prediction$`1`)
png(filename="AUC-RandomeForest.png")
plot(roc_obj, main = "ROC Curve - RandomeForest")
dev.off()
auc(roc_obj)
# feature importance
varImpRf = varImp(rf_default, scale = FALSE)
png(filename="CoefficientPlot-RandomeForest.png")
plot(varImpRf, top = 20)
dev.off()
# Bagging
levels(train$chgoff_flag) <- c('NotChargedOff','ChargedOff')
levels(test$chgoff_flag) <- c('NotChargedOff','ChargedOff')
ctrl <- trainControl(method = "repeatedcv", repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
set.seed(5627)
orig_fit <- train(chgoff_flag ~ ., data = train, 
                  method = "treebag",
                  nbagg = 50,
                  metric = "ROC",
                  trControl = ctrl)
print(orig_fit)
bag_pred <- predict(orig_fit, test)
confusionMatrix(bag_pred, test$chgoff_flag)

library(pROC)
bag_pred <- predict(orig_fit, test,type='prob')
levels(test$chgoff_flag) <- c('0','1')
roc_obj <- roc(test$chgoff_flag, bag_pred$ChargedOff)
auc(roc_obj)
########## logistic regression to pred chgoff_flag ########## 
levels(train$chgoff_flag) <- c('0','1')
logitModel <- glm(chgoff_flag ~.,family=binomial(link='logit'), data=train)
summary(logitModel)
p <- predict(logitModel, newdata=test, type="response")
roc_obj <- roc(test$chgoff_flag, p)
plot(roc_obj, main = "ROC Curve - Logistic Regression")
auc(roc_obj)

########## Penalized Logistic Regression ########
library(glmnet)
x_train <- model.matrix(chgoff_flag~., train)[,-21]
y_train <- train$chgoff_flag
x_test <- model.matrix(chgoff_flag~., test)[,-21]
# Find the best lambda using cross-validation
set.seed(123) 
cv.lasso <- cv.glmnet(x_train, y_train, type.measure="auc", alpha = 1, family = "binomial")
plot(cv.lasso)
coef(cv.lasso, s = "lambda.1se")
library(coefplot)
png(filename="CoefficientPlot-Lasso.png")
coefplot(cv.lasso, lambda=cv.lasso$lambda.1se, sort='magnitude')
dev.off()
prob_lasso = predict(cv.lasso, newx = x_test, s = "lambda.1se", type='response')
roc_obj <- roc(test$chgoff_flag, prob_lasso)
png(filename="AUC-Lasso.png")
plot(roc_obj, main = "ROC Curve - Penalized Logistic Regression Lasso")
dev.off()
auc(roc_obj)

set.seed(123) 
cv.ridge <- cv.glmnet(x_train, y_train, type.measure="auc", alpha = 0, family = "binomial")
plot(cv.ridge)
coef(cv.ridge, s = "lambda.1se")
png(filename="CoefficientPlot-Ridge.png")
coefplot(cv.ridge, lambda=cv.lasso$lambda.1se, sort='magnitude')
dev.off()
prob_ridge = predict(cv.ridge, newx = x_test, s = "lambda.1se", type='response')
roc_obj <- roc(test$chgoff_flag, prob_ridge)
plot(roc_obj, main = "ROC Curve - Penalized Logistic Regression Ridge")
auc(roc_obj)



