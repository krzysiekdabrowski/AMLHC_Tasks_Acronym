data(cox2)
X <- cox2Descr
y <- cox2Class

inTrain <- createDataPrtition(y, 0.75, list=false)

X_train <- X[inTrain,]
y_train <- y[inTrain]
X_test <- X[-inTrain,]
y_test <- y[-inTrain]

trControl <- trainControl(method="cv", number=10)
model_rf <- train(X_train, y_train,  method="rf", preProcess=c("center","scale"),
                  trainControl=trControl)

plot(varImp(model_rf))

best_rf <- model_rf$finalModel
y_predicted <- predict(best_rf, X_test)
confusionMatrix(y_predicted, y_test)
