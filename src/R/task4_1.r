heartdata <- read.csv("/Users/k.d.dabrowski/Downloads/heartdata.csv", header = TRUE, sep = "\t")

# Explore the dataset
head(heartdata)
summary(heartdata)

# Check for linearity with scatterplots
plot(heartdata$biking, heartdata$heartdisease, main = "Heart Disease vs Biking", xlab = "Biking", ylab = "Heart Disease")
plot(heartdata$smoking, heartdata$heartdisease, main = "Heart Disease vs Smoking", xlab = "Smoking", ylab = "Heart Disease")

# Fit a linear regression model
model &lt;- lm(heartdisease ~ biking + smoking, data = heartdata)

# Check for normality of residuals with histogram and Q-Q plot
hist(residuals(model), main = "Histogram of Residuals", xlab = "Residuals")
qqnorm(residuals(model))
qqline(residuals(model), col = "red")

# Check for homoscedasticity with residuals vs. fitted values plot
plot(fitted(model), residuals(model), main = "Residuals vs Fitted", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")

# Perform correlation tests to check for multicollinearity
cor.test(heartdata$biking, heartdata$smoking)

# Optional: Print the summary of the linear model
summary(model)
