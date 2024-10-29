import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from smooth_tile import predict_img_with_smooth_windowing


img_path = ''
model_path = ''
# size of patches
patch_size = 256
# Number of classes 
n_classes = 6

# Convert labeled images back to original RGB colored masks. 
def label_to_rgb(predicted_image):
    
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    
    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    Road = '#6EC1E4'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    
    Vegetation =  'FEDD3A'.lstrip('#') 
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    
    Water = 'E2A929'.lstrip('#') 
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    
    Unlabeled = '#9B9B9B'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
    
    
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)

def predict(img_path, model_path, patch_size,n_classes):

    scaler = MinMaxScaler()
    img = cv2.imread(img_path, 1)
    original_mask = cv2.imread("test_data/mask_part_008.png", 1)
    original_mask = cv2.cvtColor(original_mask,cv2.COLOR_BGR2RGB)
    # Loading the model
    model = load_model(model_path, compile=False)
                  
    #Predict using smooth blending
    input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

    # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
    # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
    predictions_smooth = predict_img_with_smooth_windowing(
        input_img,
        window_size=patch_size,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=n_classes,
        pred_func=(
            lambda img_batch_subdiv: model.predict((img_batch_subdiv))
        )
    )

    final_prediction = np.argmax(predictions_smooth, axis=2)

    return final_prediction


final_prediction = predict(img_path, model_path, patch_size,n_classes)

#Plot and save results
results=label_to_rgb(final_prediction)

plt.figure(figsize=(12, 12))
plt.title('Prediction with smooth blending')
plt.imshow(results)
plt.show()
