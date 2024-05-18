require(caret)
data(BloodBrain)

X <- bbbDescr
y <- logBBB

inTrain <- createDataPrtition(y, 0.75, list=false)

X_train <- X[inTrain,]
y_train <- y[inTrain]
X_test <- X[-inTrain,]
y_test <- y[-inTrain]
