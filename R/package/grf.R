# This function fits a geographical random forest model.
# Inputs:
# - formula: an object of class "formula" or one that can be coerced to that class
# - dframe: a data frame containing the variables in the model
# - bw: bandwidth, used for kernel density estimation
# - kernel: type of kernel to use ('adaptive' or 'fixed')
# - coords: coordinates for the geographical data
# - ntree: number of trees to grow in the forest
# - mtry: number of variables randomly sampled as candidates at each split
# - importance: type of importance measure ('impurity' or 'permutation')
# - nthreads: number of threads for parallel processing
# - forests: boolean indicating whether to save the local forests
# - geo.weighted: boolean indicating whether to use geographical weighting
# - print.results: boolean indicating whether to print the results
# - ...: additional arguments passed to the ranger function


grf <- function(formula, dframe, bw, kernel, coords, ntree=500, mtry=NULL,maxdepth = NULL, importance="impurity", nthreads = NULL, forests = TRUE, geo.weighted = TRUE,  print.results=TRUE, ...)
{

  # Start timing the function execution
  start.time <- Sys.time()

  # # Convert formula text to a formula object
  f <- formula(formula)

  # Extract variable names from the formula
  RNames <- attr(terms(f, data = dframe), "term.labels")

  # Get the name of the dependent variable
  DepVarName <- row.names(attr(terms(f, data = dframe), "factors"))[1]

  # Create a data frame for the dependent variable
  Y.DF <- dframe[DepVarName]

  # Convert the dependent variable data frame to a vector
  Y <- Y.DF[[1]]

  # Determine the number of independent variables and add 1 for degrees of freedom
  ModelVarNo <- length(RNames)
  K = ModelVarNo + 1

  # Set the number of trees in the model
  ntrees <- ntree
  maxdepths <- maxdepth

  # Set seed for reproducibility
  set.seed(42)

  # make a copy of dataset for global model
  dframe_full <- dframe

  # Use createDataPartition to create indices for random sampling
  train_indices <- createDataPartition(dframe_full$dust_storm, p = 0.8, list = FALSE)
  train.df <- dframe_full[train_indices, ]
  coords_train <- coords[train_indices, ]


  # Use createDataPartition again for the validation set
  valid.df <- dframe_full[-train_indices, ]
  coords_valid <- coords[-train_indices, ]

  # Count the number of observations in the data
  Obs <- nrow(dframe)

  # Define mtry if it is not provided [max(floor(Number of Variables/3), 1)]
  if (is.null(mtry)) {mtry= max(floor(ModelVarNo/3), 1)}

  # Print initial information if required
  if(print.results) {
    message("\nNumber of Observations: ", Obs)
    message("Number of Independent Variables: ", ModelVarNo)
  }

  # Configure the kernel type and its parameters
  if(kernel == 'adaptive')
  {
    Ne <- bw
    if(print.results) {message("Kernel: Adaptive\nNeightbours: ", Ne)}
  }
  else
  {
    if(kernel == 'fixed')
    {
      if(print.results) {message("Kernel: Fixed\nBandwidth: ", bw)}
    }
  }

  # Fit the global random forest model using the ranger package
  Gl.Model <- eval(substitute(ranger(formula, data = train.df, num.trees=ntree, mtry= mtry,max.depth=maxdepths, importance=importance, num.threads = nthreads, classification = TRUE, ...)))

  # Get predictions from the global model
  Predict <- predict(Gl.Model, valid.df, num.threads = nthreads, type = "response")

  # Predict2 <- predict(Gl.Model, dframe, num.threads = nthreads, type = "response")
  #
  # yhat2 <- Predict2$predictions
  #
  # combined_data2 <- bind_cols(yhat2, coords)
  # #
  # write.csv(combined_data2, "D:/University/DustStorming/ToAli/Geographically_weighted_random_forest/file2.csv", row.names = FALSE)
  #

  yhat <- Predict$predictions

  if (any(is.na(yhat))) stop("Missing values in predictions.")

  # Print global model summary if required
  if(print.results) {
    message("\n--------------- Global ML Model Summary ---------------\n")
    print(Gl.Model)

    message("\nImportance:\n")
    print(Gl.Model$variable.importance)

    # Convert factor variables to numeric
    Y_numeric <- as.numeric(as.character(valid.df$dust_storm))
    yhat_numeric <- as.numeric(as.character(Predict$predictions))

    # Confusion matrix
    g.conf_matrix <- table(Actual = Y_numeric, Predicted = (yhat_numeric> 0.5))

    # Classification metrics
    g.TP <- g.conf_matrix[2, 2]
    g.TN <- g.conf_matrix[1, 1]
    g.FP <- g.conf_matrix[1, 2]
    g.FN <- g.conf_matrix[2, 1]

    # Accuracy
    g.accuracy <- (g.TP + g.TN) / sum(g.conf_matrix)

    # Precision
    g.precision <- g.TP / (g.TP + g.FP)

    # Recall (Sensitivity)
    g.recall <- g.TP / (g.TP + g.FN)

    # F1-score
    g.f1_score <- 2 * (g.precision * g.recall) / (g.precision + g.recall)

    # AUC (Area Under the ROC Curve)
    g.roc <- pROC::roc(Y_numeric, yhat_numeric)
    g.auc <- g.roc$auc

    message("\nConfusion Matrix:\n")
    print(g.conf_matrix)

    message("\nAccuracy: ", round(g.accuracy, 3))
    message("Precision: ", round(g.precision, 3))
    message("Recall: ", round(g.recall, 3))
    message("F1-Score: ", round(g.f1_score, 3))
    message("AUC: ", round(g.auc, 3))
  }


  # Threshold to Drop the unimportant features for local model training
  importance_threshold <- 1.5

  # Identify columns with low importance
  low_importance_columns <- names(Gl.Model$variable.importance)[Gl.Model$variable.importance < importance_threshold]

  # Drop low-importance columns from train.df
  dframe <- dframe[, -which(names(dframe) %in% low_importance_columns)]

  RNames <- attr(terms(f, data = dframe), "term.labels")

  # Calculate distances between observations based on coordinates
  DistanceT <- dist(coords)
  Dij <- as.matrix(DistanceT)

  # Initialize storage for local forests if required
  if (forests == TRUE) {LM_Forests <- as.list(rep(NA, length(ntrees)))}

  # Empty data frame for storing local feature importance
  LM_LEst <- as.data.frame(setNames(replicate(ModelVarNo, numeric(0), simplify = F), RNames[1:ModelVarNo]))

  # Empty data frame for storing OOB Results
  LM_GofFit <- data.frame(y=numeric(0), LM_yfitOOB=numeric(0), LM_ResOOB=numeric(0), LPerm=numeric(0))

  # Empty data frame for storing prediction Results
  LM_GofFit_predict <-data.frame(y=numeric(0), LM_yfitPred=numeric(0), Diff_y_predict=numeric(0))

  # Empty data frame for storing Aggregted NEW OOB Results
  LM_Result_OOB <-data.frame(y=numeric(0), LM_yfitOOB=numeric(0), PoinID=numeric(0))

  # Empty data frame for storing Aggregted NEW Predicted Results
  LM_Result_Predict <-data.frame(y=numeric(0), LM_yfitPred=numeric(0), PoinID=numeric(0))

  for(m in 1:Obs){

    #Get the data
    DNeighbour <- Dij[,m]
    dframe$pointID <- seq_len(nrow(dframe))
    DataSet <- data.frame(dframe, DNeighbour = DNeighbour)

    #Sort by distance
    DataSetSorted <- DataSet[order(DataSet$DNeighbour),]

    if(kernel == 'adaptive')
    {
      cc <- 1
      #Keep Nearest Neighbours
      SubSet <- DataSetSorted[1:Ne,]

      # make sure there is a least one type of both labels in the subset
      while (length(unique(SubSet$dust_storm))<2){
        SubSet <- DataSetSorted[1:(Ne+cc),]
        cc <- cc +1
      }

      Kernel_H <- max(SubSet$DNeighbour)
    }
    else
    {
      if(kernel == 'fixed')
      {
        SubSet <- subset(DataSetSorted, DNeighbour <= bw)
        Kernel_H <- bw
      }
    }


    # Set seed for reproducibility
    set.seed(42)

    # create training subset data frame
    train_indices <- createDataPartition(SubSet$dust_storm, p = 0.8, list = FALSE)
    SubSet_train.df <- SubSet[train_indices, ]
    if (length(unique(SubSet$dust_storm))==2){
      # Ensure that both classes are present in the training set
      while (length(unique(SubSet_train.df$DNeighbour[train_indices])) < 2) {
        train_indices <- createDataPartition(SubSet$dust_storm, p = 0.8, list = FALSE)
        SubSet_train.df <- SubSet[train_indices, ]
      }
    }

    # calculate weights for the local model case weights
    Wts_train <- (1-(SubSet_train.df$DNeighbour/Kernel_H)^2)^2 #Bi-square weights
    # Wts_train <- exp(-(SubSet_train.df$DNeighbour^2) / (2 * Kernel_H^2)) # Gaussian weights

    # create validation subset data frame
    SubSet_valid.df <- SubSet[-train_indices, ]

    # calculate the weights for imbalanced dataset for class weights of local model
    num_class_0_SubSet_train <- sum(SubSet_train.df$dust_storm == 0)
    num_class_1_SubSet_train <- sum(SubSet_train.df$dust_storm == 1)
    num_class_total <- num_class_0_SubSet_train + num_class_1_SubSet_train

    weight_class_0_SubSet_train <- num_class_total / (2 * num_class_0_SubSet_train)
    weight_class_1_SubSet_train <- num_class_total / (2 * num_class_1_SubSet_train)

    weight_class_0_normalized <- weight_class_0_SubSet_train / (weight_class_0_SubSet_train + weight_class_1_SubSet_train)
    weight_class_1_normalized <- weight_class_1_SubSet_train / (weight_class_0_SubSet_train + weight_class_1_SubSet_train)

    # class weights of local model
    classweights_SubSet_train = c(weight_class_0_normalized,weight_class_1_normalized)


    #Calculate WLM
    if (geo.weighted == TRUE) {

      # Train a local model
      Lcl.Model <- eval(substitute(ranger(formula, data = SubSet_train.df, num.trees=ntree, mtry= mtry, importance=importance, case.weights=Wts_train,class.weights = classweights_SubSet_train ,max.depth =maxdepths, num.threads = nthreads, classification = TRUE, ...)))
      local.predicted.y <- Lcl.Model$predictions[[1]]
      counter <- 1

      while (is.nan(local.predicted.y)) {

        Lcl.Model <- eval(substitute(ranger(formula, data = SubSet_train.df, num.trees=ntree, mtry= mtry, importance=importance, case.weights=Wts_train,class.weights = classweights_SubSet_train,max.depth =maxdepths, num.threads = nthreads, classification = TRUE, ...)))

        local.predicted.y <- Lcl.Model$predictions[[1]]
        counter <- counter + 1
      }
    } else
    {
      Lcl.Model <- eval(substitute(ranger(formula, data = SubSet_train.df, num.trees=ntree, mtry= mtry, importance=importance, class.weights = classweights_SubSet_train,max.depth = maxdepths, num.threads = nthreads, classification = TRUE, ...)))
      local.predicted.y <- Lcl.Model$predictions[[1]]
      counter <- 1
    }

    if (forests == TRUE) {LM_Forests[[m]] <- Lcl.Model}

    #Importance
    for (j in 1:ModelVarNo) {
      LM_LEst[m,j] <- Lcl.Model$variable.importance[j]
    }


    # Results for OOB NEW Method
    Combine_LM_Result_OOB <- data.frame(SubSet_train.df$dust_storm,local.predicted.y, SubSet_train.df$pointID)
    LM_Result_OOB <- rbind(LM_Result_OOB, Combine_LM_Result_OOB)

    # Results for Predicted NEW Method
    l.predict <- predict(Lcl.Model, SubSet_valid.df, num.threads = nthreads)
    Combine_LM_Result_Predict <- data.frame(SubSet_valid.df$dust_storm,l.predict$predictions, SubSet_valid.df$pointID)
    LM_Result_Predict <- rbind(LM_Result_Predict, Combine_LM_Result_Predict)

    # Results for OOB Method
    LM_GofFit[m,1] <- as.numeric(as.character(SubSet_train.df$dust_storm[1]))
    LM_GofFit[m,2] <- as.numeric(as.character(Lcl.Model$predictions[1]))
    LM_GofFit[m,3] <- ifelse(LM_GofFit[m,1] == LM_GofFit[m,2], 0, 1)  # Assuming 0 for correct classification, 1 for incorrect
    LM_GofFit[m,4] <- counter

    # Results for Predicted Method
    LM_GofFit_predict[m,1] <- as.numeric(as.character(SubSet_valid.df$dust_storm[1]))
    LM_GofFit_predict[m,2] <- as.numeric(as.character(l.predict$predictions[1]))
    LM_GofFit_predict[m,3] <- LM_GofFit_predict[m,1] - LM_GofFit_predict[m,2]

  }

  # Compile outputs from the function
  if (forests == TRUE) {grf.out <- list(Global.Model=Gl.Model, Locations = Coords, Local.Variable.Importance = LM_LEst, LGofFit=LM_GofFit,LGofFitpredict =  LM_GofFit_predict , Forests=LM_Forests)}
  else {grf.out <- list(Global.Model=Gl.Model, Locations = Coords, Local.Variable.Importance = LM_LEst, LGofFit=LM_GofFit, LGofFitpredict =  LM_GofFit_predict)}
  if(print.results) {

    message("\n--------------- Local Model Summary ---------------\n")

    message("\nConfusion Matrix for OOB New Method:\n")

    # Results for OOB NEW Method # Add TP,TN, FP,FN to the dataframe
    LM_Result_OOB <- LM_Result_OOB %>%
      mutate(
        TP = ifelse(SubSet_train.df.dust_storm == 1 & local.predicted.y == 1, 1, 0),
        TN = ifelse(SubSet_train.df.dust_storm == 0 & local.predicted.y == 0, 1, 0),
        FP = ifelse(SubSet_train.df.dust_storm == 0 & local.predicted.y == 1, 1, 0),
        FN = ifelse(SubSet_train.df.dust_storm == 1 & local.predicted.y == 0, 1, 0)
      )

    # Aggregte TP,TN, FP,FN to the dataframe for every PointID
    LM_Result_OOB_summary <- LM_Result_OOB %>%
      group_by(SubSet_train.df.pointID) %>%
      summarize(
        TP_count = sum(TP),
        TN_count = sum(TN),
        FP_count = sum(FP),
        FN_count = sum(FN)
      )

    # Majority decision for TP,TN, FP,FN for every PointID
    LM_Result_OOB_summary <- LM_Result_OOB_summary %>%
      rowwise() %>%
      mutate(Classification = case_when(
        TP_count == max(TP_count, TN_count, FP_count, FN_count) & TP_count > 0 ~ "TP",
        TN_count == max(TP_count, TN_count, FP_count, FN_count) & TN_count > 0 ~ "TN",
        FP_count == max(TP_count, TN_count, FP_count, FN_count) & FP_count > 0 ~ "FP",
        FN_count == max(TP_count, TN_count, FP_count, FN_count) & FN_count > 0 ~ "FN",
        TRUE ~ "No classification"
      ))

    LM_Result_OOB_classification_counts <- table(LM_Result_OOB_summary$Classification)

    LM_Result_OOB_matrix <- matrix(c(LM_Result_OOB_classification_counts["TP"], LM_Result_OOB_classification_counts["FN"],
                                             LM_Result_OOB_classification_counts["FP"], LM_Result_OOB_classification_counts["TN"]),
                                            nrow = 2, byrow = TRUE,
                                            dimnames = list(c("Actual Positive", "Actual Negative"),
                                                            c("Predicted Positive", "Predicted Negative")))

    # Display the confusion matrix
    print(LM_Result_OOB_matrix)

    TP_OOB_N <- LM_Result_OOB_matrix[1, 1]
    TN_OOB_N <- LM_Result_OOB_matrix[2, 2]
    FP_OOB_N <- LM_Result_OOB_matrix[2, 1]
    FN_OOB_N <- LM_Result_OOB_matrix[1, 2]

    l.accuracy_OOB_N <- (TP_OOB_N + TN_OOB_N) / sum(LM_Result_OOB_matrix)
    l.precision_OOB_N <- TP_OOB_N / (TP_OOB_N + FP_OOB_N)
    l.recall_OOB_N <- TP_OOB_N / (TP_OOB_N + FN_OOB_N)
    l.f1_score_OOB_N <- 2 * (l.precision_OOB_N * l.recall_OOB_N) / (l.precision_OOB_N + l.recall_OOB_N)

    message("\nBinary Classification Metrics for OOB New Method:\n")
    message("Accuracy: ", round(l.accuracy_OOB_N, 3))
    message("Precision: ", round(l.precision_OOB_N, 3))
    message("Recall: ", round(l.recall_OOB_N, 3))
    message("F1-Score: ", round(l.f1_score_OOB_N, 3))


    message("\nConfusion Matrix for Predicted New Method:\n")

    # Results for Prediction NEW Method # Add TP,TN, FP,FN to the dataframe
    LM_Result_Predict <- LM_Result_Predict %>%
      mutate(
        TP = ifelse(SubSet_valid.df.dust_storm == 1 & l.predict.predictions == 1, 1, 0),
        TN = ifelse(SubSet_valid.df.dust_storm == 0 & l.predict.predictions == 0, 1, 0),
        FP = ifelse(SubSet_valid.df.dust_storm == 0 & l.predict.predictions == 1, 1, 0),
        FN = ifelse(SubSet_valid.df.dust_storm == 1 & l.predict.predictions == 0, 1, 0)
             )

    # Aggregte TP,TN, FP,FN to the dataframe for every PointID
    LM_Result_Predict_summary <- LM_Result_Predict %>%
      group_by(SubSet_valid.df.pointID) %>%
      summarize(
        TP_count = sum(TP),
        TN_count = sum(TN),
        FP_count = sum(FP),
        FN_count = sum(FN)
      )

    # LM_Result_Predict_summary <- LM_Result_Predict_summary %>%
    #   mutate(
    #     Predicted_Value = ifelse(TP_count > 0 | FP_count > 0, 1, 0)
    #   )

    # Majority decision for TP,TN, FP,FN for every PointID
    LM_Result_Predict_summary <- LM_Result_Predict_summary %>%
      rowwise() %>%
      mutate(Classification = case_when(
        TP_count == max(TP_count, TN_count, FP_count, FN_count) & TP_count > 0 ~ "TP",
        TN_count == max(TP_count, TN_count, FP_count, FN_count) & TN_count > 0 ~ "TN",
        FP_count == max(TP_count, TN_count, FP_count, FN_count) & FP_count > 0 ~ "FP",
        FN_count == max(TP_count, TN_count, FP_count, FN_count) & FN_count > 0 ~ "FN",
        TRUE ~ "No classification"
      ))
    # combined_data <- bind_cols(LM_Result_Predict_summary, coords)
    #
    # write.csv(combined_data, "D:/University/DustStorming/ToAli/Geographically_weighted_random_forest/file.csv", row.names = FALSE)
    #



    LM_Result_Predict_classification_counts <- table(LM_Result_Predict_summary$Classification)

    LM_Result_Predict_conf_matrix <- matrix(c(LM_Result_Predict_classification_counts["TP"], LM_Result_Predict_classification_counts["FN"],
                            LM_Result_Predict_classification_counts["FP"], LM_Result_Predict_classification_counts["TN"]),
                          nrow = 2, byrow = TRUE,
                          dimnames = list(c("Actual Positive", "Actual Negative"),
                                          c("Predicted Positive", "Predicted Negative")))

    # Display the confusion matrix
    print(LM_Result_Predict_conf_matrix)

    TP_Pred_N <- LM_Result_Predict_conf_matrix[1, 1]
    TN_Pred_N <- LM_Result_Predict_conf_matrix[2, 2]
    FP_Pred_N <- LM_Result_Predict_conf_matrix[2, 1]
    FN_Pred_N <- LM_Result_Predict_conf_matrix[1, 2]

    l.accuracy_Pred_N <- (TP_Pred_N + TN_Pred_N) / sum(LM_Result_Predict_conf_matrix)
    l.precision_Pred_N <- TP_Pred_N / (TP_Pred_N + FP_Pred_N)
    l.recall_Pred_N <- TP_Pred_N / (TP_Pred_N + FN_Pred_N)
    l.f1_score_Pred_N <- 2 * (l.precision_Pred_N * l.recall_Pred_N) / (l.precision_Pred_N + l.recall_Pred_N)


    message("\nBinary Classification for Predicted New Method:\n")
    message("Accuracy: ", round(l.accuracy_Pred_N, 3))
    message("Precision: ", round(l.precision_Pred_N, 3))
    message("Recall: ", round(l.recall_Pred_N, 3))
    message("F1-Score: ", round(l.f1_score_Pred_N, 3))

    message("\nConfusion Matrix for OOB:\n")
    confusion_matrix_OOB <- table(Actual = grf.out$LGofFit$y, Predicted = (grf.out$LGofFit$LM_yfitOOB > 0.5))
    print(confusion_matrix_OOB)

    message("\nConfusion Matrix for Predicted:\n")
    confusion_matrix_Pred <- table(Actual = grf.out$LGofFitpredict$y, Predicted = grf.out$LGofFitpredict$LM_yfitPred )
    print(confusion_matrix_Pred)


  }
  lvi <- data.frame(Min = apply(grf.out$Local.Variable.Importance, 2, min), Max = apply(grf.out$Local.Variable.Importance, 2, max),
                    Mean = apply(grf.out$Local.Variable.Importance, 2, mean), StD = apply(grf.out$Local.Variable.Importance, 2, sd))


  TP_OOB <- confusion_matrix_OOB[2, 2]
  TN_OOB <- confusion_matrix_OOB[1, 1]
  FP_OOB <- confusion_matrix_OOB[1, 2]
  FN_OOB <- confusion_matrix_OOB[2, 1]

  l.accuracy_OOB <- (TP_OOB + TN_OOB) / sum(confusion_matrix_OOB)
  l.precision_OOB <- TP_OOB / (TP_OOB + FP_OOB)
  l.recall_OOB <- TP_OOB / (TP_OOB + FN_OOB)
  l.f1_score_OOB <- 2 * (l.precision_OOB * l.recall_OOB) / (l.precision_OOB + l.recall_OOB)

  # Calculate binary classification metrics for Predicted
  TP_Pred <- confusion_matrix_Pred[2, 2]
  TN_Pred <- confusion_matrix_Pred[1, 1]
  FP_Pred <- confusion_matrix_Pred[1, 2]
  FN_Pred <- confusion_matrix_Pred[2, 1]

  l.accuracy_Pred <- (TP_Pred + TN_Pred) / sum(confusion_matrix_Pred)
  l.precision_Pred <- TP_Pred / (TP_Pred + FP_Pred)
  l.recall_Pred <- TP_Pred / (TP_Pred + FN_Pred)
  l.f1_score_Pred <- 2 * (l.precision_Pred * l.recall_Pred) / (l.precision_Pred + l.recall_Pred)

  if(print.results) {

    message("\nBinary Classification Metrics for OOB:\n")
    message("Accuracy: ", round(l.accuracy_OOB, 3))
    message("Precision: ", round(l.precision_OOB, 3))
    message("Recall: ", round(l.recall_OOB, 3))
    message("F1-Score: ", round(l.f1_score_OOB, 3))

    message("\nBinary Classification Metrics for Predicted:\n")
    message("Accuracy: ", round(l.accuracy_Pred, 3))
    message("Precision: ", round(l.precision_Pred, 3))
    message("Recall: ", round(l.recall_Pred, 3))
    message("F1-Score: ", round(l.f1_score_Pred, 3))
  }

  lModelSummary = list()
  lModelSummary$l.VariableImportance <- lvi
  lModelSummary$l.accuracy_OOB <- l.accuracy_OOB
  lModelSummary$l.precision_OOB <- l.precision_OOB
  lModelSummary$l.recall_OOB <- l.recall_OOB
  lModelSummary$l.f1_score_OOB <- l.f1_score_OOB

  grf.out$LocalModelSummary <- lModelSummary

  # Calculate and print the time taken to run the function
  end.time <- Sys.time()
  time.taken <- end.time - start.time

  if(print.results) {message("\nCalculation time (in seconds): ", round(time.taken,4))}

  # Return the output list
  return(grf.out)
}
