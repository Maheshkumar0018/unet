from utils import dice_coef, jaccard_coef
from model import UNET_model, get_UNET_model
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot, plot_model, to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
from preprocess import load_and_patchify_images_mask,plot_random_patchify_images_mask, rgb_to_2D_label


root_directory = '/home/mglocadmin/Downloads/Semantic segmentation dataset/'
patch_size = 256  
# Loading the data and performing patch
image_dataset, mask_dataset = load_and_patchify_images_mask(root_directory, patch_size)
# Plot random patch data (verification)
plot_random_patchify_images_mask(image_dataset,mask_dataset,patch_size)

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)    

labels = np.array(labels)   
labels = np.expand_dims(labels, axis=3)
n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)

# Preparing data for training
X_train, X_test, y_train, y_test = train_test_split(image_dataset, 
                                                    labels_cat, test_size = 0.20, 
                                                    random_state = 42)
image_height = X_train.shape[1]
image_width = X_train.shape[2]
n_classes = y_train.shape[3]
image_channels = X_train.shape[3]

model = get_UNET_model(n_classes=n_classes, 
                       image_height=image_height, 
                       image_width=image_width,
                       image_channels=image_channels)
model.get_config()
# Save model blue-print
SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, expand_nested=True)
# Complie
model.compile(optimizer='Adam', loss='categorical_crossentropy' , metrics=["accuracy", dice_coef])
model.summary()


#fit
unet_model_history = model.fit(X_train, y_train,
                          batch_size=8,
                          epochs=100,
                          validation_data=(X_test, y_test),
                          verbose=1,
                          shuffle=False)


model.save('./satellite-imagery.h5')