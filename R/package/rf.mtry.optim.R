rf.mtry.optim <- function(formula, data, min.mtry = NULL, max.mtry = NULL, mtry.step = 1, cv.method = "repeatedcv", cv.folds = 10, classification_metric = "AUC", ...) {

  RNames <- all.vars(formula)[-1]  # Extract variable names from the formula

  ModelVarNo <- length(RNames)

  if (is.null(min.mtry)) {
    min.mtry <- 1
  }
  if (is.null(max.mtry)) {
    max.mtry <- ModelVarNo
  }

  if (cv.method == "repeatedcv") {
    control <- trainControl(cv.method, repeats = 5, number = cv.folds, search = "grid", classProbs = TRUE, summaryFunction = twoClassSummary, ...)
  } else if (cv.method == "cv") {
    control <- trainControl(number = cv.folds, cv.method, search = "grid", classProbs = TRUE, summaryFunction = twoClassSummary, ...)
  } else {
    control <- trainControl(number = cv.folds, cv.method, search = "grid", classProbs = TRUE, summaryFunction = twoClassSummary, ...)
  }

  set.seed(123)

  tunegrid <- expand.grid(.mtry = seq(from = min.mtry, to = max.mtry, by = mtry.step))

  # Use the formula directly without conversion
  rf_gridsearch <- train(formula, data = data, method = "rf", tuneGrid = tunegrid, trControl = control, ...)

  print(rf_gridsearch)

  plot(rf_gridsearch)

  return(rf_gridsearch)
}
