#detach(training_data)
library(devtools)
library(binaryLogic)
library(MASS)
library(corrplot)
library(moments)
library(GGally)
library(ROCR)
library(tree)
library(randomForest)
library(neuralnet)
#install.packages("tensorflow")
library(tensorflow)
library(keras)
#install_tensorflow()ne
# install_github("d4ndo/binaryLogic")
# install.packages("devtools")
# library(binaryLogic)

training_data <- read.csv("advertising_train.csv", header = TRUE)
training_data$X <- NULL



# install.packages("remotes")
# remotes::install_github(sprintf("rstudio/%s", c("reticulate", "tensorflow", "keras")))
# reticulate::miniconda_uninstall() # start with a blank slate
# reticulate::install_miniconda()
# keras::install_keras()

attach(training_data)
#as.factor(Male)
#as.factor(Clicked.on.Ad)
#training_data$Male <- as.binary(training_data$Male)
#training_data$Clicked.on.Ad <- as.binary(training_data$Clicked.on.Ad)
training_data$Ad.Topic.Line <- as.factor(training_data$Ad.Topic.Line)
training_data$City <- as.factor(training_data$City)
training_data$Country <- as.factor(training_data$Country)
training_data$Timestamp <- as.factor(training_data$Timestamp)

summary(Daily.Time.Spent.on.Site)
hist(Daily.Time.Spent.on.Site, freq = FALSE, xlab = "Daily time spent on site", col = "light blue")
skewness(Daily.Time.Spent.on.Site)
kurtosis(Daily.Time.Spent.on.Site)

summary(Age)
hist(Age, freq = FALSE, xlab = "Age", col = "yellow")
skewness(Age)
kurtosis(Age)
?kurtosis

summary(Area.Income)
hist(Area.Income, freq = FALSE, xlab = "Area income", col = "red")
skewness(Area.Income)
kurtosis(Area.Income)

summary(Daily.Internet.Usage)
hist(Daily.Internet.Usage, freq = FALSE, xlab = "Daily internet usage (minutes)", col = "orange")
skewness(Daily.Internet.Usage)
kurtosis(Daily.Internet.Usage)

male_table <- table(training_data$Male)
male_perc <- male_table / length(training_data$Male)
male_perc
table(Clicked.on.Ad)


summary(Ad.Topic.Line, 5)
summary(City, 15)
summary(Country,15)



#Multivariate Analysis
training_data_quant_var <- training_data[, c(1,2,3,4)]
cor_quant.var <- cor(training_data_quant_var)
corrplot(cor_quant.var, method="color", type="upper")
cor_quant.var

Clicked.on.Ad_factorial <- as.factor(training_data$Clicked.on.Ad)
ggpairs(training_data_quant_var, aes(colour= Clicked.on.Ad_factorial))





#LOGISTIC REGRESSION
#Logistic regression con tutte le variabili
glm.fit <- glm(Clicked.on.Ad ~ Daily.Time.Spent.on.Site + Age + Area.Income + Daily.Internet.Usage + Male, family = binomial, data = training_data)
summary(glm.fit)
glm.probs <- predict(glm.fit, type = "response")
pred.glm <- rep(0, length(glm.probs))
pred.glm[glm.probs > 0.5] <- 1
table(pred.glm, Clicked.on.Ad)
conf.matrix <- addmargins(table(pred.glm, Clicked.on.Ad))
conf.matrix
mean(pred.glm != Clicked.on.Ad)
search()

#logistic regression senza la variabile Male
glm2.fit <- glm(Clicked.on.Ad ~ Daily.Time.Spent.on.Site + Age + Area.Income + Daily.Internet.Usage, family = binomial, data = training_data)
summary(glm2.fit)
glm.probs2 <- predict(glm2.fit, type = "response")
pred.glm2 <- rep(0, length(glm.probs))
pred.glm2[glm.probs2 > 0.5] <- 1
table(pred.glm2, Clicked.on.Ad)
conf.matrix <- addmargins(table(pred.glm2, Clicked.on.Ad))
conf.matrix
mean(pred.glm2 != Clicked.on.Ad)

#logistic regression con stepwise 
glm3.fit = glm(Clicked.on.Ad ~ Daily.Time.Spent.on.Site + Age + Area.Income + Daily.Internet.Usage + Male, family = binomial, data = training_data) %>% stepAIC(trace = FALSE)
glm.probs3 <- predict(glm3.fit, type = "response")
pred_glm3 <- ifelse(glm.probs3>0.5, 1, 0)
# pred_val.nn <- rep(0, length(p_val))
# pred_val.nn[p_val > 0.5] <- 1
table(pred_glm3, Clicked.on.Ad)
mean(pred_glm3 != Clicked.on.Ad)


#ROC Curve
ROCPred <- ROCR::prediction(glm.probs2, Clicked.on.Ad)
ROCPerf <-performance(ROCPred, "tpr", "fpr")
plot(ROCPerf,colorize=TRUE,lwd=2)
plot(ROCPerf,colorize=TRUE,lwd=2, print.cutoffs.at=c(0.2,0.5,0.8))
abline(a=0,b=1, lty=2)

ROCauc <-performance(ROCPred, measure ="auc")
ROCauc@y.values[[1]]





#RANDOM FORESTS
tree.training <- tree(Clicked.on.Ad_factorial ~ ., data = training_data_quant_var)
tree.training
summary(tree.training)
plot(tree.training, lwd=1,type="uniform")
text(tree.training,cex=0.75,col="blue")

set.seed(1)
rf.training <- randomForest(Clicked.on.Ad_factorial ~ ., data = training_data_quant_var, importance = TRUE)
rf.training
importance(rf.training)
varImpPlot(rf.training, col="blue", main="Training data")




#NEURAL NETWORKS
#min-max normalization
training_data_quant_var$Daily.Time.Spent.on.Site <- (training_data_quant_var$Daily.Time.Spent.on.Site - min(training_data_quant_var$Daily.Time.Spent.on.Site))/(max(training_data_quant_var$Daily.Time.Spent.on.Site) - min(training_data_quant_var$Daily.Time.Spent.on.Site))
training_data_quant_var$Age <- (training_data_quant_var$Age - min(training_data_quant_var$Age))/(max(training_data_quant_var$Age) - min(training_data_quant_var$Age))
training_data_quant_var$Area.Income <- (training_data_quant_var$Area.Income - min(training_data_quant_var$Area.Income))/(max(training_data_quant_var$Area.Income) - min(training_data_quant_var$Area.Income))
training_data_quant_var$Daily.Internet.Usage <- (training_data_quant_var$Daily.Internet.Usage - min(training_data_quant_var$Daily.Internet.Usage))/(max(training_data_quant_var$Daily.Internet.Usage) - min(training_data_quant_var$Daily.Internet.Usage))

#building neural network
set.seed(2)
n <- neuralnet(Clicked.on.Ad ~ Daily.Time.Spent.on.Site + Age + Area.Income + Daily.Internet.Usage,
                                          data = training_data_quant_var,
                                          hidden = 1,
                                          err.fct = "ce",
                                          linear.output = FALSE,
                                       #  lifesign = "full", #to see every feasible output
                                          rep = 3)
plot(n, rep = 2)




#prediction
output <- compute(n, training_data_quant_var, rep = 2)
head(output$net.result,15)
head(training_data[1,])




#confusion matrix
output <- compute(n, training_data_quant_var, rep = 2)
p1 <- output$net.result
pred.nn <- rep(0, length(p1))
pred.nn[p1 > 0.5] <- 1
table(pred.nn, Clicked.on.Ad)
mean(pred.nn != Clicked.on.Ad)





#provo con KERAS i NN
tensorflow::set_random_seed(42)
data_scale <- scale(training_data_quant_var)
trainLabels <- to_categorical(training_data$Clicked.on.Ad)
print(trainLabels)
train_NN_model <- keras_model_sequential() 
train_NN_model %>% layer_dense(units = 4, activation = 'relu', input_shape = c(4)) %>%
                  layer_dropout(rate = 0.3) %>%
                  layer_dense(units= 2, activation = "sigmoid") #because we have 2 categories in the response variable
summary(train_NN_model)
train_NN_model %>% compile(loss = 'binary_crossentropy', optimizer = optimizer_rmsprop(), metrics ="accuracy")
history <- train_NN_model %>% fit(data_scale, trainLabels, epoch = 50, batch_size = 32)
plot(history)

train_NN_model %>% evaluate(data_scale, trainLabels)

#Validation dataset
validation_dataset <- read.csv("advertising_validation.csv", header = TRUE)
validation_dataset$X <- NULL
validation_dataset$Ad.Topic.Line <- as.factor(validation_dataset$Ad.Topic.Line)
validation_dataset$City <- as.factor(validation_dataset$City)
validation_dataset$Country <- as.factor(validation_dataset$Country)
validation_dataset$Timestamp <- as.factor(validation_dataset$Timestamp)
validation_dataset_quant_var <- validation_dataset[, c(1,2,3,4)]

#prediction with logistic regression in validation set
val.probs = predict(glm2.fit, newdata=validation_dataset, type="response") 
pred.val.glm <- ifelse(val.probs > 0.5, 1, 0)
#pred.val.glm <- rep(0, length(val.probs))
#pred.val.glm[val.probs > 0.5] <- 1
table(pred.val.glm, validation_dataset$Clicked.on.Ad)
mean(pred.val.glm != validation_dataset$Clicked.on.Ad)

#prediction with random forests in validation set
rf.test = predict(rf.training, newdata=validation_dataset, type="response") 
table(rf.test, validation_dataset$Clicked.on.Ad)
mean(rf.test != validation_dataset$Clicked.on.Ad)

#prediction with NN in validation set
output_val <- compute(n, validation_dataset_quant_var, rep=2)
p_val <- output_val$net.result
pred_val.nn <- ifelse(p_val>0.5, 1, 0)
# pred_val.nn <- rep(0, length(p_val))
# pred_val.nn[p_val > 0.5] <- 1
table(pred_val.nn, validation_dataset$Clicked.on.Ad)
mean(pred_val.nn != validation_dataset$Clicked.on.Ad)

#prediction with NN CON KERAS in validation set
valLabels <- to_categorical(validation_dataset$Clicked.on.Ad)
data_scale_val <- scale(validation_dataset_quant_var)
train_NN_model %>% evaluate(data_scale_val, valLabels)
summary(train_NN_model)


#test_data
test_data <- read.csv("advertising_test.csv", header = TRUE)
test_data$X <- NULL
test.probs = predict(glm2.fit, newdata=test_data, type="response")
pred.test.glm <- ifelse(test.probs > 0.5, 1, 0)
dframe_final <- test_data
dframe_final$Clicked.on.Ad <- pred.test.glm
head(dframe_final)
write.csv(dframe_final,"advertising_test_results.csv", row.names = FALSE)


