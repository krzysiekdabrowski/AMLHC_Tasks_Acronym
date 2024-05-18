install.packages("foreign")
install.packages("caret")
library(foreign)
library(caret)

diabetes <- read.arff("diabetes.arff")
head(diabetes)

glm_model <- glm(class ~ ., data = diabetes, family = binomial)
summary(glm_model)

# Plot diagnostic plots for the GLM
par(mfrow = c(2, 2))  # Arrange plots in a 2x2 grid
plot(glm_model)
par(mfrow = c(1, 1))  # Reset to default

train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation
train_model <- train(class ~ ., 
                     data = diabetes, 
                     method = "glm", 
                     family = "binomial",
                     trControl = train_control)

print(train_model)
