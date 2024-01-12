#object = local Forests
predict.grf <- function(object, new.data, x.var.name, y.var.name, local.w = 1, global.w = 0, threshold = 0.5, ...) {

  Obs <- nrow(new.data)

  predictions <- vector(mode = "numeric", length = Obs)

  for (i in 1:Obs) {

    x <- new.data[i, which(names(new.data) == x.var.name)]
    y <- new.data[i, which(names(new.data) == y.var.name)]

    locations <- object$Locations

    D <- sqrt((x - locations[, 1])^2 + (y - locations[, 2])^2)

    local.model.ID <- which.min(D)

    g.predict <- predict(object[[1]], new.data[i, ], ...)
    g.probabilities <- g.predict$predictions
    l.predict <- predict(object$Forests[[local.model.ID]], new.data[i, ])
    l.probabilities <- l.predict$predictions

    # Update: Combine probability estimates for binary classification
    combined.probabilities <- global.w * g.probabilities + local.w * l.probabilities

    # Update: Assign binary class based on the specified threshold
    predictions[i] <- ifelse(combined.probabilities[1] >= threshold, 1, 0)
  }
  return(predictions)
}
