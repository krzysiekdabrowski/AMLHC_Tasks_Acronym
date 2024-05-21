install.packages("caret")
library(caret)

heartdata <- read.csv("/Users/k.d.dabrowski/Downloads/heartdata.csv", header = TRUE, sep = "\t")

cor.test(heartdata$biking, heartdata$smoking)
hist(residuals(model), main = "Histogram of Residuals", xlab = "Residuals")
qqnorm(residuals(model))
qqline(residuals(model), col = "red")
plot(fitted(model), residuals(model), main = "Residuals vs Fitted", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")
 

model <- lm(heartdisease ~ biking + smoking, data = heartdata)

summary(model)
par(mfrow = c(2, 2))  # Arrange plots in a 2x2 grid
plot(model)

train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

model <- train(heartdisease ~ biking + smoking, 
               data = heartdata, 
               method = "lm", 
               trControl = train_control)
print(model)

