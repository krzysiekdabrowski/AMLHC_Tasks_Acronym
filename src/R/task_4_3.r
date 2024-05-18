require(caret)
data(BloodBrain)

X <- bbbDescr
y <- logBBB

inTrain <- createDataPrtition(y, 0.75, list=false)

X_train <- X[inTrain,]
y_train <- y[inTrain]
X_test <- X[-inTrain,]
y_test <- y[-inTrain]

featVar <- apply(X_train, 2, var)
length(featVar<0.001)
featInd <- featVar>0

X_train <- Xtrain[,featVar>0]
X_test <- X_test[,featVar>0]

trControl <- trainControl(method="cv", number=10)
model_rf <- train(X_train, y_train,  method="rf", preProcess=c("center","scale"),
                  trainControl=trControl)

vi <- varImp(model_rf)
best_rf <- model_rf$finalModel

y_predicted <- predict(best_rf, X_test)
RMSE(y_predicted, y_test)
