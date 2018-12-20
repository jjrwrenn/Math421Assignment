# Clear environment
rm(list = ls())
set.seed(12192018)

# Read in data file
data = read.csv(file = "C:\\Users\\student\\Documents\\Applied Data Mining\\Final Project\\default of credit card clients.csv", header = TRUE)

# Remove this NULLs variable, is ID only
data$X<-NULL

# Replace names
names(data) <- c('credit', 'gender', 'education', 'maritalStatus', 'age', 'sepPaymentStatus', 'augPaymentStatus', 'julPaymentStatus', 'junPaymentStatus', 'mayPaymentStatus', 'aprPaymentStatus', 'sepBillStatement', 'augBillStatement', 'julBillStatement', 'junBillStatement', 'mayBillStatement', 'aprBillStatement', 'sepPrevBillStatement', 'augPrevBillStatement', 'julPrevBillStatement', 'junPrevBillStatement', 'mayPrevBillStatement', 'aprPrevBillStatement', 'target')

# Make odd values missing
data$education[data$education == 6]<-NA
data$education[data$education == 5]<-NA
data$education[data$education == 0]<-NA

data$maritalStatus[data$maritalStatus == 0]<-NA

data$sepPaymentStatus[data$sepPaymentStatus == -2]<-NA
data$augPaymentStatus[data$augPaymentStatus == -2]<-NA
data$julPaymentStatus[data$julPaymentStatus == -2]<-NA
data$junPaymentStatus[data$junPaymentStatus == -2]<-NA
data$mayPaymentStatus[data$mayPaymentStatus == -2]<-NA
data$aprPaymentStatus[data$aprPaymentStatus == -2]<-NA

data$sepBillStatement[data$sepBillStatement < 0]<-NA
data$augBillStatement[data$augBillStatement < 0]<-NA
data$julBillStatement[data$julBillStatement < 0]<-NA
data$junBillStatement[data$junBillStatement < 0]<-NA
data$mayBillStatement[data$mayBillStatement < 0]<-NA
data$aprBillStatement[data$aprBillStatement < 0]<-NA

data$sepPrevBillStatement[data$sepPrevBillStatement < 0]<-NA
data$augPrevBillStatement[data$augPrevBillStatement < 0]<-NA
data$julPrevBillStatement[data$julPrevBillStatement < 0]<-NA
data$junPrevBillStatement[data$junPrevBillStatement < 0]<-NA
data$mayPrevBillStatement[data$mayPrevBillStatement < 0]<-NA
data$aprPrevBillStatement[data$aprPrevBillStatement < 0]<-NA

# Make some ints factors
target.f <- factor(data$target, labels = c("No", "Yes"))
gender.f <- factor(data$gender, labels = c("M", "F"))
edu.f <- factor(data$education, labels = c("GradSchool", "University", "HighSchool", "Other"))
marital.f <- factor(data$maritalStatus, labels = c("Married", "Single", "Other"))

data$target <- target.f
data$gender <- gender.f
data$education <- edu.f
data$maritalStatus <- marital.f

# Under sample, will make later computations faster and will balance the data
data.no = data[data$target == 'No',]
data.yes = data[data$target == 'Yes',]
data.no.under = data.no[sample(1:nrow(data.no), nrow(data.yes)),]
data.under <- rbind(data.yes, data.no.under)

# Do imputations before binning
data.under$education[is.na(data.under$education)]<-sort(names(table(data.under$education)))[1]
data.under$maritalStatus[is.na(data.under$maritalStatus)]<-sort(names(table(data.under$maritalStatus)))[1]

# Save for later use
data.bin <- data.under
data.dummy <- data.under

# Complete imputing
data.median<-predict(preProcess(data.under, method='medianImpute'), newdata = data.under)
data.knn<-predict(preProcess(data.under, method='knnImpute'), newdata = data.under)
# Change to complete.cases() soon
data.complete<-data.under[complete.cases(data.under),]

# Model Creation: Median Impute
  # Quick Clean
median.splitIndex <- createDataPartition(data.median$target, p = 0.70, list = FALSE, times = 1)
median.train <- data.median[ median.splitIndex,]
median.test  <- data.median[-median.splitIndex,]
median.train.list <- c(median.train$credit, median.train$gender, median.train$education, median.train$maritalStatus, median.train$age, median.train$sepPaymentStatus, median.train$augPaymentStatus, median.train$julPaymentStatus, median.train$junPaymentStatus, median.train$mayPaymentStatus, median.train$aprPaymentStatus, median.train$sepBillStatement, median.train$augBillStatement, median.train$julBillStatement, median.train$junBillStatement, median.train$mayBillStatement, median.train$aprBillStatement, median.train$sepPrevBillStatement, median.train$augPrevBillStatement, median.train$julPrevBillStatement, median.train$junBillStatement, median.train$mayPrevBillStatement, median.train$aprPrevBillStatement)
median.train.matrix <- matrix(median.train.list, nrow = length(median.train.list)/23, ncol = 23)
median.test.list <- c(median.test$credit, median.test$gender, median.test$education, median.test$maritalStatus, median.test$age, median.test$sepPaymentStatus, median.test$augPaymentStatus, median.test$julPaymentStatus, median.test$junPaymentStatus, median.test$mayPaymentStatus, median.test$aprPaymentStatus, median.test$sepBillStatement, median.test$augBillStatement, median.test$julBillStatement, median.test$junBillStatement, median.test$mayBillStatement, median.test$aprBillStatement, median.test$sepPrevBillStatement, median.test$augPrevBillStatement, median.test$julPrevBillStatement, median.test$junBillStatement, median.test$mayPrevBillStatement, median.test$aprPrevBillStatement)
median.test.matrix <- matrix(median.test.list, nrow = length(median.test.list)/23, ncol = 23)

  # Glmnet model
fit.median = glmnet(x = median.train.matrix, y = as.double(data.median[median.splitIndex,]$target), family = "binomial")
fit.median.predict <- predict(fit.median, newx = median.test.matrix)

  # C5.0 model
C5.0.tree.median <- C5.0(x = median.train[.1:(ncol(median.train)-1)], y = median.train$target)
C5.0.tree.median.predict <- predict(C5.0.tree.median, newdata = median.test)

  # rpart model
median.rpart <- rpart(target ~ ., data = median.train, method = "class")
median.pred <- predict(median.rpart, median.test, type = "class")
median.cm = confusionMatrix(data = median.pred, reference = (median.test$target), positive = "Yes")

  # ranger model
myGrid7median = expand.grid(mtry = 2, splitrule = ("gini"),
                           min.node.size = c(1:3))
model7median <- train(target~.,data = median.train, method = "ranger", 
                     trControl = trainControl(method ="cv", number = 7, verboseIter = TRUE),
                     tuneGrid = myGrid7median)


# Model Creation: Knn Impute
  # Quick Clean
knn.splitIndex <- createDataPartition(data.knn$target, p = 0.70, list = FALSE, times = 1)
knn.train <- data.knn[ knn.splitIndex,]
knn.test  <- data.knn[-knn.splitIndex,]
knn.train.list <- c(knn.train$credit, knn.train$gender, knn.train$education, knn.train$maritalStatus, knn.train$age, knn.train$sepPaymentStatus, knn.train$augPaymentStatus, knn.train$julPaymentStatus, knn.train$junPaymentStatus, knn.train$mayPaymentStatus, knn.train$aprPaymentStatus, knn.train$sepBillStatement, knn.train$augBillStatement, knn.train$julBillStatement, knn.train$junBillStatement, knn.train$mayBillStatement, knn.train$aprBillStatement, knn.train$sepPrevBillStatement, knn.train$augPrevBillStatement, knn.train$julPrevBillStatement, knn.train$junBillStatement, knn.train$mayPrevBillStatement, knn.train$aprPrevBillStatement)
knn.train.matrix <- matrix(knn.train.list, nrow = length(knn.train.list)/23, ncol = 23)
knn.test.list <- c(knn.test$credit, knn.test$gender, knn.test$education, knn.test$maritalStatus, knn.test$age, knn.test$sepPaymentStatus, knn.test$augPaymentStatus, knn.test$julPaymentStatus, knn.test$junPaymentStatus, knn.test$mayPaymentStatus, knn.test$aprPaymentStatus, knn.test$sepBillStatement, knn.test$augBillStatement, knn.test$julBillStatement, knn.test$junBillStatement, knn.test$mayBillStatement, knn.test$aprBillStatement, knn.test$sepPrevBillStatement, knn.test$augPrevBillStatement, knn.test$julPrevBillStatement, knn.test$junBillStatement, knn.test$mayPrevBillStatement, knn.test$aprPrevBillStatement)
knn.test.matrix <- matrix(knn.test.list, nrow = length(knn.test.list)/23, ncol = 23)

# Glmnet model
fit.knn = glmnet(x = knn.train.matrix, y = as.double(data.knn[knn.splitIndex,]$target), family = "binomial")
fit.knn.predict <- predict(fit.knn, newx = knn.test.matrix)

# C5.0 model
C5.0.tree.knn <- C5.0(x = knn.train[.1:(ncol(knn.train)-1)], y = knn.train$target)
C5.0.tree.knn.predict <- predict(C5.0.tree.knn, newdata = knn.test)

# rpart model
knn.rpart <- rpart(target ~ ., data = knn.train, method = "class")
knn.pred <- predict(knn.rpart, knn.test, type = "class")
knn.cm = confusionMatrix(data = knn.pred, reference = (knn.test$target), positive = "Yes")

# ranger model
myGrid7knn = expand.grid(mtry = 2, splitrule = ("gini"),
                            min.node.size = c(1:3))
model7knn <- train(target~.,data = knn.train, method = "ranger", 
                      trControl = trainControl(method ="cv", number = 7, verboseIter = TRUE),
                      tuneGrid = myGrid7knn)

# Model Creation: complete Impute
  # Quick Clean
complete.splitIndex <- createDataPartition(data.complete$target, p = 0.70, list = FALSE, times = 1)
complete.train <- data.complete[ complete.splitIndex,]
complete.test  <- data.complete[-complete.splitIndex,]
complete.train.list <- c(complete.train$credit, complete.train$gender, complete.train$education, complete.train$maritalStatus, complete.train$age, complete.train$sepPaymentStatus, complete.train$augPaymentStatus, complete.train$julPaymentStatus, complete.train$junPaymentStatus, complete.train$mayPaymentStatus, complete.train$aprPaymentStatus, complete.train$sepBillStatement, complete.train$augBillStatement, complete.train$julBillStatement, complete.train$junBillStatement, complete.train$mayBillStatement, complete.train$aprBillStatement, complete.train$sepPrevBillStatement, complete.train$augPrevBillStatement, complete.train$julPrevBillStatement, complete.train$junBillStatement, complete.train$mayPrevBillStatement, complete.train$aprPrevBillStatement)
complete.train.matrix <- matrix(complete.train.list, nrow = length(complete.train.list)/23, ncol = 23)
complete.test.list <- c(complete.test$credit, complete.test$gender, complete.test$education, complete.test$maritalStatus, complete.test$age, complete.test$sepPaymentStatus, complete.test$augPaymentStatus, complete.test$julPaymentStatus, complete.test$junPaymentStatus, complete.test$mayPaymentStatus, complete.test$aprPaymentStatus, complete.test$sepBillStatement, complete.test$augBillStatement, complete.test$julBillStatement, complete.test$junBillStatement, complete.test$mayBillStatement, complete.test$aprBillStatement, complete.test$sepPrevBillStatement, complete.test$augPrevBillStatement, complete.test$julPrevBillStatement, complete.test$junBillStatement, complete.test$mayPrevBillStatement, complete.test$aprPrevBillStatement)
complete.test.matrix <- matrix(complete.test.list, nrow = length(complete.test.list)/23, ncol = 23)

# Glmnet model
fit.complete = glmnet(x = complete.train.matrix, y = as.double(data.complete[complete.splitIndex,]$target), family = "binomial")
fit.complete.predict <- predict(fit.complete, newx = complete.test.matrix)

# C5.0 model
C5.0.tree.complete <- C5.0(x = complete.train[.1:(ncol(complete.train)-1)], y = complete.train$target)
C5.0.tree.complete.predict <- predict(C5.0.tree.complete, newdata = complete.test)

# rpart model
complete.rpart <- rpart(target ~ ., data = complete.train, method = "class")
complete.pred <- predict(complete.rpart, complete.test, type = "class")
complete.cm = confusionMatrix(data = complete.pred, reference = (complete.test$target), positive = "Yes")

# ranger model
myGrid7complete = expand.grid(mtry = 2, splitrule = ("gini"),
                         min.node.size = c(1:3))
model7complete <- train(target~.,data = complete.train, method = "ranger", 
                   trControl = trainControl(method ="cv", number = 7, verboseIter = TRUE),
                   tuneGrid = myGrid7complete)


# Binning cause I'm lazy
data.bin$sepPaymentStatus[is.na(data.bin$sepPaymentStatus)]<-0
data.bin$augPaymentStatus[is.na(data.bin$augPaymentStatus)]<-0
data.bin$julPaymentStatus[is.na(data.bin$julPaymentStatus)]<-0
data.bin$junPaymentStatus[is.na(data.bin$junPaymentStatus)]<-0
data.bin$mayPaymentStatus[is.na(data.bin$mayPaymentStatus)]<-0
data.bin$aprPaymentStatus[is.na(data.bin$aprPaymentStatus)]<-0

  # Make on time or little delay (0) a 0 value, all else a 1 (late, will make a factor later)
data.bin$sepPaymentStatus[data.bin$sepPaymentStatus <= 0]<-0
data.bin$augPaymentStatus[data.bin$augPaymentStatus <= 0]<-0
data.bin$julPaymentStatus[data.bin$julPaymentStatus <= 0]<-0
data.bin$junPaymentStatus[data.bin$junPaymentStatus <= 0]<-0
data.bin$mayPaymentStatus[data.bin$mayPaymentStatus <= 0]<-0
data.bin$aprPaymentStatus[data.bin$aprPaymentStatus <= 0]<-0

data.bin$sepPaymentStatus[data.bin$sepPaymentStatus > 0]<-1
data.bin$augPaymentStatus[data.bin$augPaymentStatus > 0]<-1
data.bin$julPaymentStatus[data.bin$julPaymentStatus > 0]<-1
data.bin$junPaymentStatus[data.bin$junPaymentStatus > 0]<-1
data.bin$mayPaymentStatus[data.bin$mayPaymentStatus > 0]<-1
data.bin$aprPaymentStatus[data.bin$aprPaymentStatus > 0]<-1

# Complete imputing
data.bin.median<-predict(preProcess(data.bin, method='medianImpute'), newdata = data.bin)
data.bin.knn<-predict(preProcess(data.bin, method='knnImpute'), newdata = data.bin)
data.bin.complete<-data.bin[complete.cases(data.bin),]

# Model Creation: Median Impute
# Quick Clean
bin.median.splitIndex <- createDataPartition(data.bin.median$target, p = 0.70, list = FALSE, times = 1)
bin.median.train <- data.bin.median[ bin.median.splitIndex,]
bin.median.test  <- data.bin.median[-bin.median.splitIndex,]
bin.median.train.list <- c(bin.median.train$credit, bin.median.train$gender, bin.median.train$education, bin.median.train$maritalStatus, bin.median.train$age, bin.median.train$sepPaymentStatus, bin.median.train$augPaymentStatus, bin.median.train$julPaymentStatus, bin.median.train$junPaymentStatus, bin.median.train$mayPaymentStatus, bin.median.train$aprPaymentStatus, bin.median.train$sepBillStatement, bin.median.train$augBillStatement, bin.median.train$julBillStatement, bin.median.train$junBillStatement, bin.median.train$mayBillStatement, bin.median.train$aprBillStatement, bin.median.train$sepPrevBillStatement, bin.median.train$augPrevBillStatement, bin.median.train$julPrevBillStatement, bin.median.train$junBillStatement, bin.median.train$mayPrevBillStatement, bin.median.train$aprPrevBillStatement)
bin.median.train.matrix <- matrix(bin.median.train.list, nrow = length(bin.median.train.list)/23, ncol = 23)
bin.median.test.list <- c(bin.median.test$credit, bin.median.test$gender, bin.median.test$education, bin.median.test$maritalStatus, bin.median.test$age, bin.median.test$sepPaymentStatus, bin.median.test$augPaymentStatus, bin.median.test$julPaymentStatus, bin.median.test$junPaymentStatus, bin.median.test$mayPaymentStatus, bin.median.test$aprPaymentStatus, bin.median.test$sepBillStatement, bin.median.test$augBillStatement, bin.median.test$julBillStatement, bin.median.test$junBillStatement, bin.median.test$mayBillStatement, bin.median.test$aprBillStatement, bin.median.test$sepPrevBillStatement, bin.median.test$augPrevBillStatement, bin.median.test$julPrevBillStatement, bin.median.test$junBillStatement, bin.median.test$mayPrevBillStatement, bin.median.test$aprPrevBillStatement)
bin.median.test.matrix <- matrix(bin.median.test.list, nrow = length(bin.median.test.list)/23, ncol = 23)

# Glmnet model
fit.bin.median = glmnet(x = bin.median.train.matrix, y = as.double(data.bin.median[bin.median.splitIndex,]$target), family = "binomial")
fit.bin.median.predict <- predict(fit.bin.median, newx = bin.median.test.matrix)

# C5.0 model
bin.C5.0.tree.median <- C5.0(x = bin.median.train[.1:(ncol(bin.median.train)-1)], y = bin.median.train$target)
bin.C5.0.tree.median.predict <- predict(C5.0.tree.median, newdata = bin.median.test)

# rpart model
bin.median.rpart <- rpart(target ~ ., data = bin.median.train, method = "class")
bin.median.pred <- predict(bin.median.rpart, bin.median.test, type = "class")
bin.median.cm = confusionMatrix(data = bin.median.pred, reference = (bin.median.test$target), positive = "Yes")

# ranger model
bin.myGrid7median = expand.grid(mtry = 2, splitrule = ("gini"),
                            min.node.size = c(1:3))
bin.model7median <- train(target~.,data = bin.median.train, method = "ranger", 
                      trControl = trainControl(method ="cv", number = 7, verboseIter = TRUE),
                      tuneGrid = bin.myGrid7median)


# Model Creation: Knn Impute
# Quick Clean
bin.knn.splitIndex <- createDataPartition(data.bin.knn$target, p = 0.70, list = FALSE, times = 1)
bin.knn.train <- data.bin.knn[ bin.knn.splitIndex,]
bin.knn.test  <- data.bin.knn[-bin.knn.splitIndex,]
bin.knn.train.list <- c(bin.knn.train$credit, bin.knn.train$gender, bin.knn.train$education, bin.knn.train$maritalStatus, bin.knn.train$age, bin.knn.train$sepPaymentStatus, bin.knn.train$augPaymentStatus, bin.knn.train$julPaymentStatus, bin.knn.train$junPaymentStatus, bin.knn.train$mayPaymentStatus, bin.knn.train$aprPaymentStatus, bin.knn.train$sepBillStatement, bin.knn.train$augBillStatement, bin.knn.train$julBillStatement, bin.knn.train$junBillStatement, bin.knn.train$mayBillStatement, bin.knn.train$aprBillStatement, bin.knn.train$sepPrevBillStatement, bin.knn.train$augPrevBillStatement, bin.knn.train$julPrevBillStatement, bin.knn.train$junBillStatement, bin.knn.train$mayPrevBillStatement, bin.knn.train$aprPrevBillStatement)
bin.knn.train.matrix <- matrix(bin.knn.train.list, nrow = length(bin.knn.train.list)/23, ncol = 23)
bin.knn.test.list <- c(bin.knn.test$credit, bin.knn.test$gender, bin.knn.test$education, bin.knn.test$maritalStatus, bin.knn.test$age, bin.knn.test$sepPaymentStatus, bin.knn.test$augPaymentStatus, bin.knn.test$julPaymentStatus, bin.knn.test$junPaymentStatus, bin.knn.test$mayPaymentStatus, bin.knn.test$aprPaymentStatus, bin.knn.test$sepBillStatement, bin.knn.test$augBillStatement, bin.knn.test$julBillStatement, bin.knn.test$junBillStatement, bin.knn.test$mayBillStatement, bin.knn.test$aprBillStatement, bin.knn.test$sepPrevBillStatement, bin.knn.test$augPrevBillStatement, bin.knn.test$julPrevBillStatement, bin.knn.test$junBillStatement, bin.knn.test$mayPrevBillStatement, bin.knn.test$aprPrevBillStatement)
bin.knn.test.matrix <- matrix(bin.knn.test.list, nrow = length(bin.knn.test.list)/23, ncol = 23)

# Glmnet model
bin.fit.knn = glmnet(x = bin.knn.train.matrix, y = as.double(data.knn[bin.knn.splitIndex,]$target), family = "binomial")
bin.fit.knn.predict <- predict(bin.fit.knn, newx = bin.knn.test.matrix)

# C5.0 model
bin.C5.0.tree.knn <- C5.0(x = bin.knn.train[.1:(ncol(bin.knn.train)-1)], y = bin.knn.train$target)
bin.C5.0.tree.knn.predict <- predict(bin.C5.0.tree.knn, newdata = bin.knn.test)

# rpart model
bin.knn.rpart <- rpart(target ~ ., data = bin.knn.train, method = "class")
bin.knn.pred <- predict(bin.knn.rpart, bin.knn.test, type = "class")
bin.knn.cm = confusionMatrix(data = bin.knn.pred, reference = (bin.knn.test$target), positive = "Yes")

# ranger model
bin.myGrid7knn = expand.grid(mtry = 2, splitrule = ("gini"),
                         min.node.size = c(1:3))
bin.model7knn <- train(target~.,data = bin.knn.train, method = "ranger", 
                   trControl = trainControl(method ="cv", number = 7, verboseIter = TRUE),
                   tuneGrid = bin.myGrid7knn)

# Model Creation: Complete Cases
# Quick Clean
bin.complete.splitIndex <- createDataPartition(data.bin.complete$target, p = 0.70, list = FALSE, times = 1)
bin.complete.train <- data.bin.complete[ bin.complete.splitIndex,]
bin.complete.test  <- data.bin.complete[-bin.complete.splitIndex,]
bin.complete.train.list <- c(bin.complete.train$credit, bin.complete.train$gender, bin.complete.train$education, bin.complete.train$maritalStatus, bin.complete.train$age, bin.complete.train$sepPaymentStatus, bin.complete.train$augPaymentStatus, bin.complete.train$julPaymentStatus, bin.complete.train$junPaymentStatus, bin.complete.train$mayPaymentStatus, bin.complete.train$aprPaymentStatus, bin.complete.train$sepBillStatement, bin.complete.train$augBillStatement, bin.complete.train$julBillStatement, bin.complete.train$junBillStatement, bin.complete.train$mayBillStatement, bin.complete.train$aprBillStatement, bin.complete.train$sepPrevBillStatement, bin.complete.train$augPrevBillStatement, bin.complete.train$julPrevBillStatement, bin.complete.train$junBillStatement, bin.complete.train$mayPrevBillStatement, bin.complete.train$aprPrevBillStatement)
bin.complete.train.matrix <- matrix(bin.complete.train.list, nrow = length(bin.complete.train.list)/23, ncol = 23)
bin.complete.test.list <- c(bin.complete.test$credit, bin.complete.test$gender, bin.complete.test$education, bin.complete.test$maritalStatus, bin.complete.test$age, bin.complete.test$sepPaymentStatus, bin.complete.test$augPaymentStatus, bin.complete.test$julPaymentStatus, bin.complete.test$junPaymentStatus, bin.complete.test$mayPaymentStatus, bin.complete.test$aprPaymentStatus, bin.complete.test$sepBillStatement, bin.complete.test$augBillStatement, bin.complete.test$julBillStatement, bin.complete.test$junBillStatement, bin.complete.test$mayBillStatement, bin.complete.test$aprBillStatement, bin.complete.test$sepPrevBillStatement, bin.complete.test$augPrevBillStatement, bin.complete.test$julPrevBillStatement, bin.complete.test$junBillStatement, bin.complete.test$mayPrevBillStatement, bin.complete.test$aprPrevBillStatement)
bin.complete.test.matrix <- matrix(bin.complete.test.list, nrow = length(bin.complete.test.list)/23, ncol = 23)

# Glmnet model
bin.fit.complete = glmnet(x = bin.complete.train.matrix, y = as.double(data.bin.complete[bin.complete.splitIndex,]$target), family = "binomial")
bin.fit.complete.predict <- predict(bin.fit.complete, newx = bin.complete.test.matrix)

# C5.0 model
bin.C5.0.tree.complete <- C5.0(x = bin.complete.train[.1:(ncol(bin.complete.train)-1)], y = bin.complete.train$target)
bin.C5.0.tree.complete.predict <- predict(bin.C5.0.tree.complete, newdata = bin.complete.test)

# rpart model
bin.complete.rpart <- rpart(target ~ ., data = bin.complete.train, method = "class")
bin.complete.pred <- predict(bin.complete.rpart, bin.complete.test, type = "class")
bin.complete.cm = confusionMatrix(data = bin.complete.pred, reference = (bin.complete.test$target), positive = "Yes")

# ranger model
bin.myGrid7complete = expand.grid(mtry = 2, splitrule = ("gini"),
                         min.node.size = c(1:3))
bin.model7complete <- train(target~.,data = bin.complete.train, method = "ranger", 
                   trControl = trainControl(method ="cv", number = 7, verboseIter = TRUE),
                   tuneGrid = bin.myGrid7complete)


# Dummy
dummies_model <- dummyVars(target~., data = data.dummy)
dummies.train <- predict(dummies_model, newdata = data.dummy)
dummies.data <- data.frame(dummies.train)
dummies.data$target <- data.dummy$target
dummies.data <- dummies.data[complete.cases(dummies.data),]

dummy.splitIndex <- createDataPartition(dummies.data$target, p = 0.70, list = FALSE, times = 1)
dummy.train <- dummies.data[ dummy.splitIndex,]
dummy.test  <- dummies.data[-dummy.splitIndex,]
dummy.train.list <- c(dummy.train$credit, dummy.train$gender.M, dummy.train$gender.F, dummy.train$education.GradSchool, dummy.train$education.University, dummy.train$education.HighSchool, dummy.train$education.Other, dummy.train$maritalStatus.Married, dummy.train$maritalStatus.Single, dummy.train$maritalStatus.Single, dummy.train$age, dummy.train$sepPaymentStatus, dummy.train$augPaymentStatus, dummy.train$julPaymentStatus, dummy.train$junPaymentStatus, dummy.train$mayPaymentStatus, dummy.train$aprPaymentStatus, dummy.train$sepBillStatement, dummy.train$augBillStatement, dummy.train$julBillStatement, dummy.train$junBillStatement, dummy.train$mayBillStatement, dummy.train$aprBillStatement, dummy.train$sepPrevBillStatement, dummy.train$augPrevBillStatement, dummy.train$julPrevBillStatement, dummy.train$junBillStatement, dummy.train$mayPrevBillStatement, dummy.train$aprPrevBillStatement)
dummy.train.matrix <- matrix(dummy.train.list, nrow = length(dummy.train.list)/29, ncol = 29)
dummy.test.list <- c(dummy.test$credit, dummy.test$gender.M, dummy.test$gender.F, dummy.test$education.GradSchool, dummy.test$education.University, dummy.test$education.HighSchool, dummy.test$education.Other, dummy.test$maritalStatus.Married, dummy.test$maritalStatus.Single, dummy.test$maritalStatus.Other, dummy.test$age, dummy.test$sepPaymentStatus, dummy.test$augPaymentStatus, dummy.test$julPaymentStatus, dummy.test$junPaymentStatus, dummy.test$mayPaymentStatus, dummy.test$aprPaymentStatus, dummy.test$sepBillStatement, dummy.test$augBillStatement, dummy.test$julBillStatement, dummy.test$junBillStatement, dummy.test$mayBillStatement, dummy.test$aprBillStatement, dummy.test$sepPrevBillStatement, dummy.test$augPrevBillStatement, dummy.test$julPrevBillStatement, dummy.test$junBillStatement, dummy.test$mayPrevBillStatement, dummy.test$aprPrevBillStatement)
dummy.test.matrix <- matrix(dummy.test.list, nrow = length(dummy.test.list)/29, ncol = 29)

# Glmnet model
fit.dummy = glmnet(x = dummy.train.matrix, y = as.double(dummies.data[dummy.splitIndex,]$target), family = "binomial")
fit.dummy.predict <- predict(fit.dummy, newx = dummy.test.matrix)

# C5.0 model
C5.0.tree.dummy <- C5.0(x = dummy.train[.1:(ncol(dummy.train)-1)], y = dummy.train$target)
C5.0.tree.dummy.predict <- predict(C5.0.tree.dummy, newdata = dummy.test)

# rpart model
dummy.rpart <- rpart(target ~ ., data = dummy.train, method = "class")
dummy.pred <- predict(dummy.rpart, dummy.test, type = "class")
dummy.cm = confusionMatrix(data = dummy.pred, reference = (dummy.test$target), positive = "Yes")

# ranger model
myGrid7dummy = expand.grid(mtry = 2, splitrule = ("gini"),
                              min.node.size = c(1:3))
model7dummy <- train(target~.,data = dummy.train, method = "ranger", 
                        trControl = trainControl(method ="cv", number = 7, verboseIter = TRUE),
                        tuneGrid = myGrid7dummy)


reportResults = function() {
  plot(fit.median.predict)
  
  plot(fit.knn.predict)
  
  plot(fit.complete.predict)

  plot(fit.bin.median.predict)
  
  plot(bin.fit.knn.predict)
  
  plot(bin.fit.complete.predict)
  
  plot(fit.dummy.predict)
  
  median.cm
  
  knn.cm
  
  complete.cm
  
  bin.median.cm
  
  bin.knn.cm
  
  bin.complete.cm
  
  dummy.cm
  
  summary(C5.0.tree.median)
  plot(C5.0.tree.median)
  
  summary(C5.0.tree.knn)
  plot(C5.0.tree.knn)
  
  summary(C5.0.tree.complete)
  plot(C5.0.tree.complete)
  
  summary(bin.C5.0.tree.median)
  plot(bin.C5.0.tree.median)
  
  summary(bin.C5.0.tree.knn)
  plot(bin.C5.0.tree.knn)
  
  summary(bin.C5.0.tree.complete)
  plot(bin.C5.0.tree.complete)
  
  summary(C5.0.tree.dummy)
  plot(C5.0.tree.dummy)
  
  model7median[[4]][c(2:4,6)]
  model7knn[[4]][c(2:4,6)]
  model7complete[[4]][c(2:4,6)]
  bin.model7median[[4]][c(2:4,6)]
  bin.model7knn[[4]][c(2:4,6)]
  bin.model7complete[[4]][c(2:4,6)]
  model7dummy[[4]][c(2:4,6)]
}

# Tuning models
fit.median.tune = glmnet(x = median.train.matrix, y = as.double(data.median[median.splitIndex,]$target), family = "binomial", standardize = FALSE)
fit.median.tune.predict <- predict(fit.median.tune, newx = median.test.matrix)
plot(fit.median.tune.predict)
plot(fit.median.tune)
coef(fit.median.tune, s = 0.01)

fit.knn.tune = glmnet(x = knn.train.matrix, y = as.double(data.knn[knn.splitIndex,]$target), family = "binomial", standardize = FALSE)
fit.knn.tune.predict <- predict(fit.knn.tune, newx = knn.test.matrix)
plot(fit.knn.tune.predict)
plot(fit.knn.tune)
coef(fit.knn.tune, s = 0.01)

fit.complete.tune = glmnet(x = knn.train.matrix, y = as.double(data.knn[knn.splitIndex,]$target), family = "binomial", standardize = FALSE)
fit.complete.tune.predict <- predict(fit.knn.tune, newx = complete.test.matrix)
plot(fit.complete.tune.predict)
plot(fit.complete.tune)
coef(fit.complete.tune, s = 0.01)

bin.fit.median.tune = glmnet(x = bin.median.train.matrix, y = as.double(data.bin.median[median.splitIndex,]$target), family = "binomial", standardize = FALSE)
bin.fit.median.tune.predict <- predict(bin.fit.median.tune, newx = bin.median.test.matrix)
plot(fit.median.tune.predict)
plot(bin.fit.median.tune)
coef(bin.fit.median.tune, s = 0.01)

bin.fit.knn.tune = glmnet(x = knn.train.matrix, y = as.double(data.bin.knn[knn.splitIndex,]$target), family = "binomial", standardize = FALSE)
bin.fit.knn.tune.predict <- predict(bin.fit.knn.tune, newx = bin.knn.test.matrix)
plot(bin.fit.knn.tune.predict)
plot(bin.fit.knn.tune)
coef(bin.fit.knn.tune, s = 0.01)

bin.fit.complete.tune = glmnet(x = bin.knn.train.matrix, y = as.double(data.bin.knn[knn.splitIndex,]$target), family = "binomial", standardize = FALSE)
bin.fit.complete.predict.tune <- predict(bin.fit.knn.tune, newx = bin.complete.test.matrix)
plot(bin.fit.complete.predict.tune)
plot(bin.fit.complete.tune)
coef(bin.fit.complete.tune, s = 0.01)

fit.dummy.tune = glmnet(x = dummy.train.matrix, y = as.double(dummies.data[dummy.splitIndex,]$target), family = "binomial", standardize = FALSE)
fit.dummy.predict.tune <- predict(fit.dummy.tune, newx = dummy.test.matrix)
plot(fit.dummy.predict.tune)
plot(fit.dummy.tune)
coef(bin.fit.complete.tune, s = 0.01)

C5.0.tree.median.tune <- C5.0(x = median.train[.1:(ncol(median.train)-1)], y = median.train$target, trials = 10)
C5.0.tree.median.predict.tune <- predict(C5.0.tree.median.tune, newdata = median.test, type = "class")

C5.0.tree.knn.tune <- C5.0(x = knn.train[.1:(ncol(knn.train)-1)], y = knn.train$target, trials = 10)
C5.0.tree.knn.predict.tune <- predict(C5.0.tree.knn.tune, newdata = knn.test, type = "class")

C5.0.tree.complete.tune <- C5.0(x = complete.train[.1:(ncol(complete.train)-1)], y = complete.train$target, trials = 10)
C5.0.tree.complete.predict.tune <- predict(C5.0.tree.complete.tune, newdata = complete.test, type = "class")

bin.C5.0.tree.median.tune <- C5.0(x = bin.median.train[.1:(ncol(bin.median.train)-1)], y = bin.median.train$target, trials = 10)
bin.C5.0.tree.median.predict.tune <- predict(bin.C5.0.tree.median.tune, newdata = bin.median.test, type = "class")

bin.C5.0.tree.knn.tune <- C5.0(x = bin.knn.train[.1:(ncol(knn.train)-1)], y = bin.knn.train$target, trials = 10)
bin.C5.0.tree.knn.predict.tune <- predict(bin.C5.0.tree.knn.tune, newdata = bin.knn.test, type = "class")

bin.C5.0.tree.complete.tune <- C5.0(x = bin.complete.train[.1:(ncol(bin.complete.train)-1)], y = bin.complete.train$target, trials = 10)
bin.C5.0.tree.complete.predict.tune <- predict(bin.C5.0.tree.complete.tune, newdata = bin.complete.test, type = "class")

C5.0.tree.dummy.tune <- C5.0(x = dummy.train[.1:(ncol(dummy.train)-1)], y = dummy.train$target, trials = 10)
C5.0.tree.dummy.predict.tune <- predict(C5.0.tree.dummy.tune, newdata = dummy.test, type = "class")

median.rpart.tune <- rpart(target ~ ., data = median.train, method = "class", control = rpart.control(xval = 20))
median.pred.tune <- predict(median.rpart.tune, median.test, type = "class")
median.cm.tune = confusionMatrix(data = median.pred.tune, reference = (median.test$target), positive = "Yes")

knn.rpart.tune <- rpart(target ~ ., data = knn.train, method = "class")
knn.pred.tune <- predict(knn.rpart.tune, knn.test, type = "class")
knn.cm.tune = confusionMatrix(data = knn.pred.tune, reference = (knn.test$target), positive = "Yes")

complete.rpart.tune <- rpart(target ~ ., data = complete.train, method = "class", control = rpart.control(xval = 20))
complete.pred.tune <- predict(complete.rpart.tune, complete.test, type = "class")
complete.cm.tune = confusionMatrix(data = complete.pred.tune, reference = (complete.test$target), positive = "Yes")

bin.median.rpart.tune <- rpart(target ~ ., data = bin.median.train, method = "class", control = rpart.control(xval = 20))
bin.median.pred.tune <- predict(bin.median.rpart.tune, bin.median.test, type = "class")
bin.median.cm.tune = confusionMatrix(data = bin.median.pred.tune, reference = (bin.median.test$target), positive = "Yes")

bin.knn.rpart.tune <- rpart(target ~ ., data = bin.knn.train, method = "class", control = rpart.control(xval = 20))
bin.knn.pred.tune <- predict(bin.knn.rpart.tune, bin.knn.test, type = "class")
bin.knn.cm.tune = confusionMatrix(data = bin.knn.pred.tune, reference = (bin.knn.test$target), positive = "Yes")

bin.complete.rpart.tune <- rpart(target ~ ., data = bin.complete.train, method = "class", control = rpart.control(xval = 20))
bin.complete.pred.tune <- predict(bin.complete.rpart.tune, bin.complete.test, type = "class")
bin.complete.cm.tune = confusionMatrix(data = bin.complete.pred.tune, reference = (bin.complete.test$target), positive = "Yes")

dummy.rpart.tune <- rpart(target ~ ., data = dummy.train, method = "class", control = rpart.control(xval = 20))
dummy.pred.tune <- predict(dummy.rpart.tune, dummy.test, type = "class")
dummy.cm.tune = confusionMatrix(data = dummy.pred.tune, reference = (dummy.test$target), positive = "Yes")

myGrid7median.tune = expand.grid(mtry = 2, splitrule = c("gini", "extratrees"),
                            min.node.size = c(1:3))
model7median.tune <- train(target~.,data = median.train, method = "ranger", 
                      trControl = trainControl(method ="cv", number = 10, verboseIter = TRUE),
                      tuneGrid = myGrid7median.tune)

myGrid7knn.tune = expand.grid(mtry = 2, splitrule = c("gini", "extratrees"),
                            min.node.size = c(1:3))
model7knn.tune <- train(target~.,data = knn.train, method = "ranger", 
                      trControl = trainControl(method ="cv", number = 10, verboseIter = TRUE),
                      tuneGrid = myGrid7knn.tune)

myGrid7complete.tune = expand.grid(mtry = 2, splitrule = c("gini", "extratrees"),
                            min.node.size = c(1:3))
model7complete.tune <- train(target~.,data = complete.train, method = "ranger", 
                      trControl = trainControl(method ="cv", number = 10, verboseIter = TRUE),
                      tuneGrid = myGrid7complete.tune)

bin.myGrid7median.tune = expand.grid(mtry = 2, splitrule = c("gini", "extratrees"),
                            min.node.size = c(1:3))
bin.model7median.tune <- train(target~.,data = bin.median.train, method = "ranger", 
                      trControl = trainControl(method ="cv", number = 10, verboseIter = TRUE),
                      tuneGrid = bin.myGrid7median.tune)

bin.myGrid7knn.tune = expand.grid(mtry = 2, splitrule = c("gini", "extratrees"),
                         min.node.size = c(1:3))
bin.model7knn.tune <- train(target~.,data = bin.knn.train, method = "ranger", 
                   trControl = trainControl(method ="cv", number = 10, verboseIter = TRUE),
                   tuneGrid = bin.myGrid7knn.tune)

bin.myGrid7complete.tune = expand.grid(mtry = 2, splitrule = c("gini", "extratrees"),
                              min.node.size = c(1:3))
bin.model7complete.tune <- train(target~.,data = bin.complete.train, method = "ranger", 
                        trControl = trainControl(method ="cv", number = 10, verboseIter = TRUE),
                        tuneGrid = bin.myGrid7complete.tune)

myGrid7dummy.tune = expand.grid(mtry = 2, splitrule = c("gini", "extratrees"),
                           min.node.size = c(1:3))
model7dummy.tune <- train(target~.,data = dummy.train, method = "ranger", 
                     trControl = trainControl(method ="cv", number = 10, verboseIter = TRUE),
                     tuneGrid = myGrid7dummy.tune)

reportResultsTune = function() {
  plot(fit.median.tune.predict)
  
  plot(fit.knn.tune.predict)
  
  plot(fit.complete.tune.predict)
  
  plot(bin.fit.median.tune.predict)
  
  plot(bin.fit.knn.tune.predict)
  
  plot(bin.fit.complete.predict.tune)
  
  plot(fit.dummy.predict.tune)
  
  median.cm.tune
  
  knn.cm.tune
  
  complete.cm.tune
  
  bin.median.cm.tune
  
  bin.knn.cm.tune
  
  bin.complete.cm.tune
  
  dummy.cm.tune
  
  summary(C5.0.tree.median.tune)
  plot(C5.0.tree.median.tune)
  
  summary(C5.0.tree.knn.tune)
  plot(C5.0.tree.knn.tune)
  
  summary(C5.0.tree.complete.tune)
  plot(C5.0.tree.complete.tune)
  
  summary(bin.C5.0.tree.median.tune)
  plot(bin.C5.0.tree.median.tune)
  
  summary(bin.C5.0.tree.knn.tune)
  plot(bin.C5.0.tree.knn.tune)
  
  summary(bin.C5.0.tree.complete.tune)
  plot(bin.C5.0.tree.complete.tune)
  
  summary(C5.0.tree.dummy.tune)
  plot(C5.0.tree.dummy.tune)
  
  model7median.tune[[4]][c(2:4,6)]
  model7knn.tune[[4]][c(2:4,6)]
  model7complete.tune[[4]][c(2:4,6)]
  bin.model7median.tune[[4]][c(2:4,6)]
  bin.model7knn.tune[[4]][c(2:4,6)]
  bin.model7complete.tune[[4]][c(2:4,6)]
  model7dummy.tune[[4]][c(2:4,6)]
}