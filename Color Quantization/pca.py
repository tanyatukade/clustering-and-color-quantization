from sklearn.decomposition import PCA
import cv2
from pathlib import Path

dir = Path(__file__).parent.absolute()
print("Path: ",dir)

# get absolute path for image input 

output = str(dir)+"/compressedImages/"
print(" Output : ", output)

for i in range(1, 4):
    
    image = cv2.imread(str(dir)+"/Input Images/image"+str(i)+".jpg")
    # Output for components = 45
    pca = PCA(n_components=45)
    rows = image.shape[0]
    columns = image.shape[1]
    channels = image.shape[2]
    print(image.shape)
    reshaped_image = image.reshape(rows, columns*channels)

    compressedImage = pca.fit_transform(reshaped_image)
    reconstructedImage = pca.inverse_transform(compressedImage)

    cv2.imwrite(output+"image"+str(i)+".jpg", reconstructedImage.reshape(image.shape))
