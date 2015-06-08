library("FNN")

train = read.csv("train.csv", header = TRUE)
train = train[, -1]
# scale predictor variables
train[, -length(train)] = scale(train[, -length(train)])

train.index = sample(nrow(train), floor(nrow(train) * 0.8))
trainset = train[train.index, ]
testset = train[-train.index, ]

n = nrow(trainset)
folds = vector(mode = "list", length = 5)
for (i in 1:5) {
  folds[[i]] = seq(i, n, by = 5)
}

# GENERATE RESPONSE
Y = trainset[, length(train)]

# Design Matrix
X = trainset[, -length(train)]

K = c(1:30)
k_num = length(K)
knn_rate = matrix(0, 5, k_num)

for(j in 1:5){
  i.tr = unlist(folds[-j])
  i.val = folds[[j]]
  X.tr = X[i.tr, ]    
  Y.tr = Y[i.tr]   
  X.val = X[i.val, ] 
  Y.val = Y[i.val]
  
  for(k in K){
    knn_fit = knn(X.tr, X.val, Y.tr, k)
    knn_output = table(knn_fit, Y.val)
    mis_class_rate_knn = (sum(knn_output) - sum(diag(knn_output)))/sum(knn_output)
    knn_rate[j,k] = mis_class_rate_knn
  }
}

save(knn_rate, file = "knn_rate.Rdata")

cv_knn = colMeans(knn_rate)

# best k
best_k = which(cv_knn == min(cv_knn))

# plot best k
plot(K, cv_knn, ylab = "Test Classification Error Rate", xlab = "K", 
     main = "Error Rate Vs. K (k-NN)")
lines(K, cv_knn)
abline(min(cv_knn), 0, col = "red")

# final k-NN model
knn_best_11 = knn(trainset[, -length(train)], testset[, -length(train)], trainset[, length(train)], k = best_k)
knn_output_11 = table(knn_best_11, testset[, length(train)])
mis_class_rate_knn_11 = (sum(knn_output_11) - sum(diag(knn_output_11)))/sum(knn_output_11)
mis_class_rate_knn_11

knn_best_5 = knn(trainset[, -length(train)], testset[, -length(train)], trainset[, length(train)], k = 5)
knn_output_5 = table(knn_best_5, testset[, length(train)])
mis_class_rate_knn_5 = (sum(knn_output_5) - sum(diag(knn_output_5)))/sum(knn_output_5)
mis_class_rate_knn_5
```


# SVM

```{r, eval=FALSE}
## SVM
library("e1071")

# linear SVM
tune.out.linear = tune.svm(target ~., data = trainset, cost = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4), 
                           kernel = 'linear', tunecontrol = tune.control(cross = 5))

best.model.linear = tune.out.linear$best.model
pred.linear = predict(best.model.linear, testset[, -length(train)])
svm.linear.output = table(predict = pred.linear, truth = testset[, length(train)])
mis_class_rate_svm_linear = (sum(svm.linear.output) - 
                               sum(diag(svm.linear.output)))/sum(svm.linear.output)
mis_class_rate_svm_linear

# radial SVM

tune.out.radial = tune.svm(target ~., data = trainset, gamma = c(0.001, 0.01, 0.1), 
                           cost = c(5, 10, 15, 30, 35, 40), kernel = 'radial', degree = 2,
                           tunecontrol = tune.control(cross = 5))

best.model.radial = tune.out.radial$best.model
pred.radial = predict(best.model.radial, testset[, -length(train)])
svm.radial.output = table(predict = pred.radial, truth = testset[, length(train)])
mis_class_rate_svm_radial = (sum(svm.radial.output) - 
                               sum(diag(svm.radial.output)))/sum(svm.radial.output)
mis_class_rate_svm_radial

# polynomial SVM

tune.out.poly = tune.svm(target ~., data = trainset, gamma = c(0.001, 0.01, 0.1, 1), 
                         cost = c(10, 15, 0.3, 0.5, 0.0001, 0.0015), kernel = 'polynomial', 
                         degree = 2, tunecontrol = tune.control(cross = 5))

best.model.poly = tune.out.poly$best.model
pred.poly = predict(best.model.poly, testset[, -length(train)])
svm.poly.output = table(predict = pred.poly, truth = testset[, length(train)])
mis_class_rate_svm_poly = (sum(svm.poly.output) - 
                             sum(diag(svm.poly.output)))/sum(svm.poly.output)
mis_class_rate_svm_poly