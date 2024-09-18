###### charged off risk ######
####### classification Task ######
setwd("/Users/shiqi/Downloads/Predictive Modeler")
library(tidyverse)
library(dplyr)
library("readxl")
data <- read_excel("Lending Analtyics.xlsx", sheet = "Dataset")

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
set.seed(345)
sample = sample.split(trimmed_data$chgoff_flag, SplitRatio = .8)
train = subset(trimmed_data, sample == TRUE)
test  = subset(trimmed_data, sample == FALSE)

#### Random Forest for Classification #########
# reset the classification name
levels(train$chgoff_flag) <- c('NotChargedOff','ChargedOff')
levels(test$chgoff_flag) <- c('NotChargedOff','ChargedOff')
# tuning mtry
trControl <- trainControl(method = "cv",number = 10, search = "grid", classProbs = TRUE) # 10 fold cross validation
tunegrid <- expand.grid(.mtry=c(10,20,30,40,50,60))
set.seed(1234)
rf_gridsearch <- train(chgoff_flag~., data=train, method="rf", metric = "ROC", tuneGrid=tunegrid, trControl=trControl)
print(rf_gridsearch) # best mtry = 10
png(filename="Random Forest Tuning-Flag.png")
plot(rf_gridsearch, main = 'Randome Forest Hyperparameter Tuning')
dev.off()

# random forest result
prediction <-predict(rf_gridsearch, test)
confusionMatrix(prediction, test$chgoff_flag)
# ROC
prediction <- predict(rf_gridsearch, test, type='prob')
roc_rf <- roc(test$chgoff_flag, prediction$ChargedOff)
png(filename="AUC-RandomeForest-Flag.png")
plot(roc_rf, main = "ROC Curve - RandomeForest")
dev.off()
auc(roc_rf)
# feature importance
varImpRf = varImp(rf_gridsearch)
png(filename="CoefficientPlot-RandomeForest-Flag.png")
plot(varImpRf, top = 20, main = 'Variable Importance Random Forest Top 20')
dev.off()

### logistic 
levels(train$chgoff_flag) <- c('0','1')
logitModel <- glm(chgoff_flag ~.,family=binomial(link='logit'), data=train)
summary(logitModel)
# plot(logitModel)
# ROC
prediction <- predict(logitModel, test, type='response')
roc_logit <- roc(test$chgoff_flag, prediction)
png(filename="AUC-Logistic-Flag.png")
plot(roc_logit, main = "ROC Curve - Logistic Regression")
dev.off()
auc(roc_logit)

coefplot(logitModel, sort='magnitude')

### logistic with lasso and elasticNet
### scaling the predictors
library(glmnet)
# tuning hyper-parameters - elasticNet
set.seed(1)

tuneGrid=expand.grid(
  .alpha= seq(0, 1, by = 0.1),
  .lambda=seq(0, 1, by = 0.1))

elastic_tune <- train(
  chgoff_flag ~ .,
  data = train,
  method = 'glmnet',
  preProcess = c("center", "scale"),
  trControl = trControl,
  tuneGrid = tuneGrid,
  family="binomial"
)
print(elastic_tune)
# 
png(filename="ElasticNet-Flag-Tuning.png")
plot(elastic_tune, main = 'ElasticNet Hyperparameter Tuning')
dev.off()
# best alpha = 0 and lambda = 0.1.
# ROC
prediction <- predict(elastic_tune, test, type='prob')
roc_elastic <- roc(test$chgoff_flag, prediction$ChargedOff)
png(filename="AUC-Logistic Elastic Net-Flag.png")
plot(roc_elastic, main = "ROC Curve - Logistic Regression Elastic Net")
dev.off()
auc(roc_elastic)


# tuning hyper-parameters- lasso
set.seed(1)

tuneGrid=expand.grid(
  .alpha= 1,
  .lambda=0.0048) #  seq(0, 1, by = 0.0001)

lasso_tune <- train(
  chgoff_flag ~ .,
  data = train,
  method = 'glmnet',
  preProcess = c("center", "scale"),
  trControl = trControl,
  tuneGrid = tuneGrid,
  family="binomial"
)
print(lasso_tune)
# best lambda = 0.0048.
# 
png(filename="Lasso-Flag-Tuning.png")
plot(lasso_tune, main = 'Lasso Hyperparameter Tuning')
dev.off()
# ROC
prediction <- predict(lasso_tune, test, type='prob')
roc_lasso <- roc(test$chgoff_flag, prediction$ChargedOff)
png(filename="AUC-Logistic Lasso-Flag.png")
plot(roc_lasso, main = "ROC Curve - Logistic Regression Lasso")
dev.off()
auc(roc_lasso)
# library(coefplot)
png(filename="CoefficientPlot2-Lasso-Flag.png.png")
coefplot(lasso_tune$finalModel, sort='magnitude', title ='Coefficient Plot for Lasso - Flag')
dev.off()

# coeficient plot
coef(lasso_tune$finalModel, lasso_tune$bestTune$lambda)

varImpRf = varImp(object=lasso_tune,scale=FALSE)
png(filename="CoefficientPlot-Lasso-Flag Not Scaled.png")
plot(varImpRf, top = 20, main = 'Variable Importance Lasso - Flag')
dev.off()

# tuning hyper-parameters - ridge
set.seed(1)

tuneGrid=expand.grid(
  .alpha= 0,
  .lambda=seq(0, 1, by =  0.0001))

ridge_tune <- train(
  chgoff_flag ~ .,
  data = train,
  method = 'glmnet',
  preProcess = c("center", "scale"),
  trControl = trControl,
  tuneGrid = tuneGrid,
  family="binomial"
)
print(ridge_tune)
# best alpha = 0 and lambda = 0.1259.
# ROC
prediction <- predict(ridge_tune, test, type='prob')
roc_ridge <- roc(test$chgoff_flag, prediction$ChargedOff)
png(filename="AUC-Logistic Ridge-Flag.png")
plot(roc_ridge, main = "ROC Curve - Logistic Regression Ridge")
dev.off()
auc(roc_ridge)

# xgboost
library(xgboost)
# one hot encoding for categorical variables
x_train <- sparse.model.matrix(chgoff_flag ~ ., data = train)[,-21]
y_train = train$chgoff_flag
x_test = sparse.model.matrix(chgoff_flag ~ ., data = test)[,-21]
y_test = test$chgoff_flag
# Specify cross-validation method and number of folds. Also enable parallel computation
xgb_trcontrol = trainControl(
  method = "cv",
  number = 10,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE, 
  classProbs = TRUE
)
# Tuning hyper-parameters
xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)
set.seed(0) 
xgb_model = train(
  x_train, y_train,  
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  method = "xgbTree"
)
print(xgb_model)
# nrounds = 100, max_depth = 20, eta = 0.1, gamma = 0, colsample_bytree = 0.8, min_child_weight = 1 and subsample = 1.
png(filename="XGBoosting-Flag-Tuning.png")
plot(xgb_model, main = 'XGBoosting Hyperparameter Tuning')
dev.off()

varImpRf = varImp(object=xgb_model)
png(filename="CoefficientPlot-XGBoostingTune-Flag.png")
plot(varImpRf, top = 20, main = 'Variable Importance XGBoosting Top 20')
dev.off()

prediction <- predict(ridge_tune, test, type='prob')
roc_xgb <- roc(test$chgoff_flag, prediction$ChargedOff)
png(filename="AUC-Logistic XGBoost-Flag.png")
plot(roc_xgb, main = "ROC Curve - XG Boosting")
dev.off()
auc(roc_xgb)

