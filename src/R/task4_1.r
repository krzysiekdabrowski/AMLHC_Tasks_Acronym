heartdata <- read.csv("/Users/k.d.dabrowski/Downloads/heartdata.csv", header = TRUE, sep = "\t")

cor.test(heartdata$biking, heartdata$smoking)

# Check for normality of residuals with histogram and Q-Q plot
hist(residuals(model), main = "Histogram of Residuals", xlab = "Residuals")
qqnorm(residuals(model))
qqline(residuals(model), col = "red")

# Check for homoscedasticity with residuals vs. fitted values plot
plot(fitted(model), residuals(model), main = "Residuals vs Fitted", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")



model <- lm(heartdisease ~ biking + smoking, data = heartdata)
summary(model)
