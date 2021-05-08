library(keras)
library(gtable)
library(scales)
library(tibble)
library(rinat)
library(sf)
library(dplyr)

# download functions from canvas #### 

source("download_images.R")
gb_11 <- readRDS("gb_simple.RDS")

# search records for white rose and download images ####

white_rose <- get_inat_obs(taxon_name = "Rosa X alba",
                           bounds = gb_11,
                           quality = "research",
                           year = 2018,
                           maxresults = 500)



download_images(spp_recs = white_rose, spp_folder = "white_rose")



# search records for midland hawthorn and download images ####


midland_hawthorn <- get_inat_obs(taxon_name = "Crataegus kaevigata",
                                 bounds = gb_11,
                                 quality = "research",
                                 year = 2018,
                                 maxresults = 500)


download_images(spp_recs = midland_hawthorn, spp_folder = "hawthorn")

# search records for dog rose and download images ####

dog_rose <- get_inat_obs(taxon_name = "Rosa canina",
                         bounds = gb_11,
                         quality = "research",
                         maxresults = 500)

download_images(spp_recs = dog_rose, spp_folder = "dog_rose")

# put the images into a seperate folder ####

image_files_path <- "images"

# list of spp to model ####

spp_list <- dir(image_files_path)

# number of spp classes ####

output_n <- length(spp_list)

# Create test, and species sub-folders ####

for(folder in 1:output_n){
  dir.create(paste("test", spp_list[folder], sep="/"), recursive=TRUE)
}

# copy images over and then delete photos from orginal folder ####

for(folder in 1:output_n){
  for(image in 401:500){
    src_image  <- paste0("images/", spp_list[folder], "/spp_", image, ".jpg")
    dest_image <- paste0("test/"  , spp_list[folder], "/spp_", image, ".jpg")
    file.copy(src_image, dest_image)
    file.remove(src_image)
  }
}

# image size to scale down to (original images vary but about 400 x 500 px) ####

img_width <- 150
img_height <- 150
target_size <- c(img_width, img_height)

# Full-colour Red Green Blue = 3 channels ####

channels <- 3

# Rescale from 255 to between zero and 1 ####

train_data_gen = image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

# now we need to train and validate the images ####

# train the images images ####

train_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "training",
                                                    seed = 42)

# validate the images ####

valid_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "validation",
                                                    seed = 42)

# check everything has been read in correctly ####

cat("Number of images per class:")

table(factor(train_image_array_gen$classes))

cat("Class labels vs index mapping")

train_image_array_gen$class_indices

# now its time to define additional parametres ####

# number of training samples ####

train_samples <- train_image_array_gen$n

# number of validation samples ####

valid_samples <- valid_image_array_gen$n

# define the batch size and number of epochs ####

batch_size <- 32

epochs <- 10

# initialise model ####

model <- keras_model_sequential()

# add layers ####

model %>%
 layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, channels), activation = "relu") %>%
  layer_conv_2d(filter = 16, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(output_n, activation = "softmax")

# check the model ####

print(model)

# compile the model ####

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# Train the model with fit_generator, first training data, then epochs, validation data then print progress ####

history <- model %>% fit_generator(
  train_image_array_gen,
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs,
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  verbose = 2
)

plot(history)

# save model for future use ####

detach("package:imager", unload = TRUE)

save.image("animals.RData")

# time to test the model with a data set that has never encountered the model, the previously created 'test' images ####

path_test <- "test"

test_data_gen <- image_data_generator(rescale = 1/255)

test_image_array_gen <- flow_images_from_directory(path_test,
 test_data_gen,
   target_size = target_size,
    class_mode = "categorical",
     classes = spp_list,
      shuffle = FALSE,
       batch_size = 1,  
        seed = 123)

# run the test model ####

model %>% evaluate_generator(test_image_array_gen, 
                             steps = test_image_array_gen$n)

# accuracy of 56% which is only 2% poorer than the training and validation dataset ####

