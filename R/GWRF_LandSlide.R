install.packages("caret")
install.packages("pROC")
install.packages("rsample")
install.packages("h2o")


library(caret)
library(rsample)
library(dplyr)
library(ranger)
library(h2o)



# Set the working directory to the package folder
setwd("D:/University/DustStorming/ToAli/DustStormModeling/R/package")

# Source each file
source("grf.R")
source("grf.bw.R")
source("predict.grf.R")
source("random.test.data.R")
source("rf.mtry.optim.R")

dataFolder<-"D:\\University\\DustStorming\\ToAli\\DustStormModeling\\R\\"
# df<-read.csv(paste0(dataFolder,"df_dustsources_WS0_X_0_PN20_SP_.csv"), header=T)
df<-read.csv(paste0(dataFolder,"Landslide_label_new.csv"), header=T)

# check if there is any NA values
df <- na.omit(df)

# Columns to convert to factors (e.g., lakes, dust_storm, dummy variables)
columns_to_convert <- c(4:20) # df_dustsources_WS0_X_0_PN20_SP_
# columns_to_convert <- c(3, 24, 27:35) # df_dustsources_WS7_X_7_PN20_SP_WMe_
# columns_to_convert <- c(4,6,8,22, 41, 45:49) # df_dustsources_WS7_X_7_PN20_SP_Var_Med_Ent_Mod

# Convert specified columns to factors
df[columns_to_convert] <- lapply(df[columns_to_convert], as.factor)

# Drop Year Column
# df <- df[, !colnames(df) %in% 'Year']

train_valid.df <- df

## Local Random Forest
Coords <- train_valid.df[, c('X_utm','Y_utm')]

# Drop the specified coordinates from the train data frame
train_valid.df <- train_valid.df[, !(colnames(train_valid.df) %in% c('X_utm','Y_utm','FID'))]

############################################################
##################### Scale the data #######################
############################################################

# Assuming columns_to_scale is a vector of column names you want to scale
# columns_to_scale <- c("Soil_evaporation", "Lakes", "Precipitation", "Soil_moisture",
#                        "NDVI", "Elevation", "Aspect", "Curvature", "Plan_curvature",
#                        "Profile_curvature", "Distance_to_river", "Slope")
#
# Ensure selected columns are numeric
# train_valid.df[, columns_to_scale] <- lapply(train_valid.df[, columns_to_scale], as.numeric)

# Extract the columns you want to scale
# columns_to_scale_data <- train_valid.df[, columns_to_scale, drop = FALSE]

# Scale the selected columns
# scaled_columns <- scale(columns_to_scale_data)

# Replace the original columns with the scaled ones
# train_valid.df[, columns_to_scale] <- scaled_columns

############################################################
###################### Hyper tuning ########################
############################################################

# h2o.init(nthreads = -1,max_mem_size ="48g",enable_assertions = FALSE)
# # Convert the data to an H2O frame
# train_valid.h2o <- as.h2o(train_valid.df)
#
# # Set the seed for reproducibility
# set.seed(123)
#
# #Generate random indices for splitting
# split_indices <- h2o.runif(train_valid.h2o, 123)
#
# # Define the ratio for splitting (e.g., 80% training, 20% validation)
# split_ratio <- 0.8
#
# # Create a mask for training set
# train_mask <- split_indices < split_ratio
#
# # Extract rows for training set
# train.h2o <- train_valid.h2o[train_mask, ,]
#
# # Extract rows for validation set
# valid.h2o <- train_valid.h2o[!train_mask, ,]
#
# # Set up the hyperparameter grid
# hyper_params <- list(
#   # ntrees = seq(900, 2000, by = 30),
#   ntrees = c(375, 399, 899),
#   mtries = c(7:15),
#   max_depth = c(8:12)
# )
#
#
#
# # Train the Random Forest model using H2O grid search
# rf_grid <- h2o.grid(
#   algorithm = "randomForest",
#   grid_id = "rf_grid",
#   x = colnames(train.h2o)[!colnames(train.h2o) %in% c("dust_storm")],
#   y = "dust_storm",
#   training_frame = train.h2o,
#   validation_frame = valid.h2o,
#   hyper_params = hyper_params,
#   nfolds=10,
#   stopping_metric = "AUC",
#   seed = 42
# )
#
# # Get the best model from the grid search
# best_rf <- h2o.getModel(rf_grid@model_ids[[1]])
#
# # Extract the best hyperparameters
# best_ntrees <- best_rf@parameters$ntrees
# best_mtries <- best_rf@parameters$mtries
# best_max_depth <- best_rf@parameters$max_depth
#
# # Stop the H2O cluster
# h2o.shutdown()

# best_ntrees <- 399
# best_mtries <- 7
# best_max_depth <- 7


############################################################
####################### GWRF Model #########################
############################################################

model <- grf(
  formula = Target ~ .,
  dframe=train_valid.df,
  bw= 140,
  ntree= 890,
  mtry = 5,
  maxdepth = 8,
  kernel="adaptive",
  forests = TRUE,
  geo.weighted = TRUE,
  importance = "impurity",
  coords=Coords
)

model$Global.Model$variable.importance

library(ggplot2)
importance_data <- data.frame(
  Variable = names(model$Global.Model$variable.importance),
  Importance = model$Global.Model$variable.importance
)

ggplot(importance_data, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Feature Importance Plot", x = "Variable", y = "Importance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

library(leaflet)
library(dplyr)

model$Locations$index <- seq_len(nrow(model$Locations))
model$Local.Variable.Importance$index <- seq_len(nrow(model$Local.Variable.Importance))

# Merge data frames based on the artificial index
merged_data <- merge(model$Locations, model$Local.Variable.Importance, by = "index")


# Define a color palette for variable importance levels
color_palette <- colorQuantile("plasma", merged_data$Slope, reverse = TRUE)

initial_view <- c(mean(merged_data$Y), mean(merged_data$X), 5.6)  # Adjust the values as needed

# Set the base map opacity
base_map_opacity <- 0.5  # Adjust the opacity as needed (0 for fully transparent, 1 for fully opaque)

# Create a leaflet map
mymap <- leaflet(merged_data, options = leafletOptions(
  title = "Your Map Title Here",
  subtitle = "Subtitle Here",
  baseMapOptions = list(opacity = base_map_opacity)
)) %>%
  setView(lng = initial_view[2], lat = initial_view[1], zoom = initial_view[3]) %>%
  addTiles() %>%  # Add a default tile layer
  addCircleMarkers(
    lng = ~X,
    lat = ~Y,
    radius = 2,  # Adjust the radius of the markers
    fillOpacity = 0.8,
    fillColor = ~color_palette(Slope),
    color = ~NA,  # Set color to NA for no border
    label = ~paste("Variable Importance: ", Slope)
  ) %>%
  addLegend(
    "bottomright",
    pal = color_palette,
    values = ~Slope,
    title = "Variable Importance",
    opacity = 1
  )

# Display the map
mymap
