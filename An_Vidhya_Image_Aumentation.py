#   https://www.analyticsvidhya.com/blog/2019/12/image-augmentation-deep-learning-pytorch/?utm_source=linkedin&utm_medium=social-media-DS&utm_campaign=DS-marketing

# IMAGE AUGMENTATION FOR DL USING PYTORCH- FEATURE ENGINEERING FOR IMAGES

# Download dataset:   https://drive.google.com/file/d/1EbVifjP0FQkyB1axb7KQ26yPtWmneApJ/view

# 1. Importar librerias
#%%
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt
#% matplotlib inline
#%%
# 2. Importar imágenes y visualizarlas
# Descargar la imagen utilizada en:  https://drive.google.com/file/d/1Ld4gDh-XjEZiCmQjvFV1WFBgC6LyvNB0/view?usp=sharing
image = io.imread('/home/jose/Documentos/ImageAugmentation/emergency_vs_non-emergency_dataset/images/0.jpg')
print(image.shape)
io.imshow(image)
#%%
# 3. Cambiar imágenes
# 3.1. Rotated image
print('Rotated Image')
rotated = rotate(image, angle=45, mode='wrap')
io.imshow(rotated)
#%%
# 3.2. Shifted image
transform = AffineTransform(translation=(25,25)) # (25,25) pixels
wrapShift = warp(image,transform, mode='wrap')
plt.imshow(wrapShift)
plt.title('Wrap Shift')
#%%
# 3.3 Flipped Image (left to right)
flipLR = np.fliplr(image)
plt.imshow(flipLR)
plt.title('Left to Right flipple')
#%%
# 3.4 Flipped Up to Down
flipUD = np.flipud(image)
plt.imshow(flipUD)
plt.title('Up Down flipped')
#%%
# 3.5. Adding Noise to Images
sigma = 0.155
noisyRandom = random_noise(image, var=sigma**2)
plt.imshow(noisyRandom)
plt.title('Random Noise')
#%%
# 3.6. Blurring images
blurred = gaussian(image, sigma=1, multichannel=True)
plt.imshow(blurred)
plt.title('Blurred image')
#%%
#----------------------------------------
# CASE STUDY
	# 1. Importing libraries and loading dataset
from torchsummary import summary
import pandas as pd
import numpy as np
from matplotlib.pyplot import imread, imsave
from tqdm import tqdm
import matplotlib.pyplot as plt
#% matplotlib inline # For Jupyter Notebook

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
#%%
data = pd.read_csv('/home/jose/Documentos/ImageAugmentation/emergency_vs_non-emergency_dataset/emergency_train.csv')
data.head()
#%%
# 2. Loading images
train_img = []
for img_name in tqdm(data['image_names']):
    image_path = '/home/jose/Documentos/ImageAugmentation/emergency_vs_non-emergency_dataset/images/' + img_name
    img = imread(image_path)
    img = img/255
    train_img.append(img)

train_x = np.array(train_img)
train_y = data['emergency_or_not'].values
train_x.shape, train_y.shape
#%%
# 3. Train
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 13, stratify=train_y)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
#%%
# 3. Augmenting the images ( se generan 4 imágenes aumentadas para cada una de las 1481 del training set) 
final_train_data = []
final_target_train = []
for i in tqdm(range(train_x.shape[0])):
    final_train_data.append(train_x[i])
    final_train_data.append(rotate(train_x[i], angle=45, mode = 'wrap'))
    final_train_data.append(np.fliplr(train_x[i]))
    final_train_data.append(np.flipud(train_x[i]))
    final_train_data.append(random_noise(train_x[i],var=0.2**2))
    for j in range(5):
        final_target_train.append(train_y[i])
#%%
# 4. Convert to an array
print(len(final_target_train), len(final_train_data))
final_train = np.array(final_train_data)
final_target_train = np.array(final_target_train)
#%%
# 5. Visualize these augmented images
fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(20,20))
for i in range(5):
    ax[i].imshow(final_train[i+30])
    ax[i].axis('off')
#%%   
# 6. Defining the DL model Architecture
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
#%%

final_train = final_train.reshape(7405, 3, 224, 224)
final_train  = torch.from_numpy(final_train)
final_train = final_train.float()

# converting the target into torch format
final_target_train = final_target_train.astype(int)
final_target_train = torch.from_numpy(final_target_train)
#%%
# Convert validation images too
val_x =val_x.reshape(165, 3, 224, 224)
val_x = torch.from_numpy(val_x)
val_x = val_x.float()

# 7. MODEL ARCHITECTURE
#%%
torch.manual_seed(0)

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.25),
            # Defining another 2D convolution layer
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.25),
            # Defining another 2D convolution layer
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.25),
            # Defining another 2D convolution layer
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.25),
        )

        self.linear_layers = Sequential(
            Linear(128 * 14 * 14, 512),
            ReLU(inplace=True),
            Dropout(),
            Linear(512, 256),
            ReLU(inplace=True),
            Dropout(),
            Linear(256,10),
            ReLU(inplace=True),
            Dropout(),
            Linear(10,2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
#%%
# Defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.000075)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
print(model)
#%%
# 8. TRAINING THE MODEL
torch.manual_seed(0)

# batch size of the model
batch_size = 32

# number of epochs to train the model
n_epochs = 20

for epoch in range(1, n_epochs+1):

    train_loss = 0.0
        
    permutation = torch.randperm(final_train.size()[0])

    training_loss = []
    for i in tqdm(range(0,final_train.size()[0], batch_size)):

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = final_train[indices], final_target_train[indices]
        
        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        
        optimizer.zero_grad()
        outputs = model.forward(batch_x)
        loss = criterion(outputs,batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
  

#%%
# 9. SAVE THE MODEL
torch.save(model, 'model.pt')
#%%
# 10. CHECKING OUR MODEL's PERFORMANCE
torch.manual_seed(0)
# prediction for training set
prediction = []
target = []
permutation = torch.randperm(final_train.size()[0])
for i in tqdm(range(0,final_train.size()[0], batch_size)):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = final_train[indices], final_target_train[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x.cuda())

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y)
#%%    
# training accuracy
accuracy = []
for i in range(len(prediction)):
    accuracy.append(accuracy_score(target[i].cpu(),prediction[i]))
    
print('training accuracy: \t', np.average(accuracy))
#%%
# 11. CHECKING THE PERFORMANCE OF VALIDATION SET

torch.manual_seed(0)
with torch.no_grad():
    output = model(val_x.cuda())
softmax = torch.exp(output).cpu()
prob = list(softmax.detach().numpy())
predictions = np.argmax(prob, axis=1)
accuracy_score(val_y, predictions)

# %%
