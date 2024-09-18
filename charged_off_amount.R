###### charged off amount ######
setwd("/Users/shiqi/Downloads/Predictive Modeler")
library(tidyverse)
library(dplyr)
library("readxl")
data <- read_excel("Lending Analtyics PM Assessment 2022.xlsx", sheet = "Dataset")
# clean the data: filtering
data_amt = filter(data, chgoff_flag == 1)
# charge off amount by random forest
library(randomForest)
require(caTools)
summary(data_amt)
sapply(data_amt, class)
colnames(data_amt)
data_amt <- transform(data_amt, Times_DQ_L12 = as.numeric(Times_DQ_L12),
                  Origination_LTV = as.numeric(Origination_LTV),
                  Direct_Deposit_flag = as.factor(Direct_Deposit_flag),
                  New_Customer_flag = as.factor(New_Customer_flag),
                  Never_DQ_flag = as.factor(Never_DQ_flag),
                  modification_flag = as.factor(modification_flag))
# Times_DQ_L12 NA to 0 times be delinquent
data_amt[["Times_DQ_L12"]][is.na(data_amt[["Times_DQ_L12"]])] <- 0
# impute origination LTV
trimmed_data = data_amt %>% select(Collateral_code,APR,orig_term,loan_amount,
                               Balance,Payment_amount,Days_Past_Due,Payments_remaining,
                               Sched_Payment_Amt,term,Num_Payments_Made,Amount_Past_Due,
                               Times_DQ_L12,Product,TotalDeposit_Amount,Customership_Tenure,DQ_Status,Origination_LTV,
                               Lending_Product_Count,Customer_Cohort,chgoff_amt,current_MOB,Tier,geo_region,
                               Generation,firstTime_borrower,Never_DQ_flag,
                               modification_flag) 
colSums(is.na(trimmed_data))
trimmed_data$Origination_LTV[is.na(trimmed_data$Origination_LTV)] <- median(trimmed_data$Origination_LTV, na.rm = T) 
# character to factor
trimmed_data <- as.data.frame(unclass(trimmed_data), stringsAsFactors = TRUE)
sapply(trimmed_data, class)
# normalize the chgoff_amt
trimmed_data['chgoff_amt'] <- trimmed_data['chgoff_amt'] / trimmed_data$loan_amount
summary(trimmed_data$chgoff_amt)
# randome forest package and tuning hyperparameters
library(caret)
set.seed(345)
sample = sample.split(trimmed_data$chgoff_amt, SplitRatio = .8)
train = subset(trimmed_data, sample == TRUE)
test  = subset(trimmed_data, sample == FALSE)

# use 10-fold cross validation
trControl <- trainControl(method = "cv",number = 10, search = "grid")
set.seed(1234)
# Run the model
rf_default <- train(chgoff_amt~.,data = train, method = "rf", trControl = trControl)
# Print the results
print(rf_default)
plot(rf_default)
# best mtry = 33
varImpRf = varImp(object=rf_default)
png(filename="CoefficientPlot-RandomeForest-AMT.png")
plot(varImpRf, top = 20, main = 'Variable Importance Random Forest Top 20')
dev.off()
# evaluating the prediction in test 
rf_pred = predict(rf_default,test)
rmse <- sqrt(sum((rf_pred - test$chgoff_amt)^2)/length(test$chgoff_amt))
rmse

# tuning mtry
tunegrid <- expand.grid(.mtry=c(10,20,30,40,50,60))
set.seed(1234)
rf_gridsearch <- train(chgoff_amt~., data=train, method="rf", tuneGrid=tunegrid, trControl=trControl)
print(rf_gridsearch)
png(filename="Random Forest Tuning-AMT.png")
plot(rf_gridsearch, main = 'Randome Forest Hyperparameter Tuning')
dev.off()
# best mtry = 40
varImpRf = varImp(object=rf_gridsearch)
png(filename="CoefficientPlot-RandomeForest-AMT.png")
plot(varImpRf, top = 20, main = 'Variable Importance Random Forest Top 20')
dev.off()
# evaluating the prediction in test 
rf_gridsearch_pred = predict(rf_gridsearch,test)
rmse <- sqrt(sum((rf_gridsearch_pred - test$chgoff_amt)^2)/length(test$chgoff_amt))
rmse

# Generalized Linear Model Regression with Lasso and Ridge
# Note that we are scaling the predictors
set.seed(123)
glmnet_model <- train(chgoff_amt ~ ., data = train, 
                      preProcess = c("center", "scale"),
                      method = "glmnet", trControl = trControl)
# Caret tested many values of the tuning parameters so we limit output
arrange(glmnet_model$results, RMSE) %>% head
# best parameter
glmnet_model$bestTune
# alpha = 1 and lambda = 0.003275807
print(glmnet_model)
# plot the coefficient
library(coefplot)
png(filename="Elastic_Net Tuning AMT.png", width=1000, height=700)
plot(glmnet_model, title = 'ElasticNet Hyperparameter Tuning')
dev.off()
# coef(glmnet_model$finalModel)
png(filename="CoefficientPlot-Elastic_Net-AMT.png", width=600, height=700)
coefplot(glmnet_model$finalModel, sort='magnitude', title ='Coefficient Plot for ElasticNet')
dev.off()
# evaluating the prediction in test 
glmnet_pred = predict(glmnet_model,test)
rmse <- sqrt(sum((glmnet_pred - test$chgoff_amt)^2)/length(test$chgoff_amt))
rmse

# Generalized Linear Model Regression with Lasso for illustration
# Note that we are scaling the predictors
set.seed(123)
ctrl <- trainControl(method = "cv",number = 10)
glmlasso_model <- train(chgoff_amt ~ ., data = train, 
                      preProcess = c("center", "scale"),
                      method = "lasso", trControl = ctrl)
# Caret tested many values of the tuning parameters so we limit output
arrange(glmlasso_model$results, RMSE) %>% head
# best parameter
glmlasso_model$bestTune # fraction = 0.5
print(glmlasso_model)
# lasso (alpha=1)
lasso_pred = predict(glmlasso_model,test)
rmse <- sqrt(sum((lasso_pred - test$chgoff_amt)^2)/length(test$chgoff_amt))
rmse

png(filename="Lasso-AMT.png")
plot(glmlasso_model)
dev.off()

# tuning hyper-parameters
set.seed(1)

tuneGrid <- expand.grid(
  .fraction = seq(0, 1, by = 0.1)
)


lasso_tune <- train(
  chgoff_amt ~ .,
  data = train,
  method = 'lasso',
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = tuneGrid
)
print(lasso_tune) # best fraction = 0.3

png(filename="Lasso-AMT-Tuning.png")
plot(lasso_tune, main = 'Lasso Hyperparameter Tuning')
dev.off()

# performance
lasso_tune_pred = predict(lasso_tune,test)
rmse <- sqrt(sum((lasso_tune_pred - test$chgoff_amt)^2)/length(test$chgoff_amt))
rmse
# coeficient plot
print(lasso_tune) # fraction = 0.3
predict(lasso_tune$finalModel,type="coef") # s = 31
predict(lasso_tune$finalModel,type="coef",s=31)
lasso_coef <- predict(lasso_tune$finalModel,type="coef",s=31)$coefficients
lasso_coef_sel <- lasso_coef[lasso_coef != 0]
plot(lasso_coef_sel)

# importance
png(filename="CoefficientPlot-LassoTune-AMT.png")
plot(varImp(lasso_tune), main = 'Variable Importance Lasso - Amount')
dev.off()

# OLS
ols_model = lm(chgoff_amt~.,data=train)
summary(ols_model)
png(filename="OLS-Residuals-AMT.png")
par(mfrow=c(2,2))
plot(ols_model)
dev.off()

ols_pred <- predict(ols_model,newdata=test)
rmse <- sqrt(sum((ols_pred - test$chgoff_amt)^2)/length(test$chgoff_amt))
rmse

#plot predicted vs. actual values
png(filename="OLS-Predicted-AMT.png")
plot(ols_pred,                                # Draw plot using Base R
     test$chgoff_amt,
     main = 'OLS Predicted vs Observed Values',
     xlab = "Predicted Values",
     ylab = "Observed Values")
abline(a = 0,                                        # Add straight line
       b = 1,
       col = "red",
       lwd = 2)
dev.off()


coefplot(ols_model, sort='magnitude')

# XG Boosting
library(xgboost)
# one hot encoding for categorical variables
x_train <- sparse.model.matrix(chgoff_amt ~ ., data = train)[,-21]
y_train = train$chgoff_amt
x_test = sparse.model.matrix(chgoff_amt ~ ., data = test)[,-21]
y_test = test$chgoff_amt
# Specify cross-validation method and number of folds. Also enable parallel computation
xgb_trcontrol = trainControl(
  method = "cv",
  number = 10,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE
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
# nrounds = 200, max_depth = 10, eta = 0.1, gamma = 0, colsample_bytree = 0.7, min_child_weight = 1 and subsample = 1.
png(filename="XGBoosting-AMT-Tuning.png")
plot(xgb_model, main = 'XGBoosting Hyperparameter Tuning')
dev.off()

varImpRf = varImp(object=xgb_model)
png(filename="CoefficientPlot-XGBoostingTune-AMT.png")
plot(varImpRf, top = 20, main = 'Variable Importance XGBoosting Top 20')
dev.off()

xgb_model$bestTune
xgb_pred <- predict(xgb_model,newdata=x_test)
rmse <- sqrt(sum((xgb_pred - test$chgoff_amt)^2)/length(test$chgoff_amt))
rmse
