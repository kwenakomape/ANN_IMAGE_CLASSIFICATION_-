from tkinter import *
from PIL import ImageTk, Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms





# Parameters for our network
LearningRate = 0.001
InputSize ,HiddenLayersSize,classes= 784,500,10
epochs,batch_size =10,100   # we going to divide 60 000 samples to 100 samples of batch

# Configure the Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset from the current directory

TrainDataset = torchvision.datasets.MNIST(root="data", 
                                           train=True, 
                                           transform=transforms.ToTensor())

TestDataset  = torchvision.datasets.MNIST(root="data", 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Load dataset to Data loader
TrainLoader = torch.utils.data.DataLoader(dataset=TrainDataset,batch_size=100, 
                                           shuffle=True)

TestLoader = torch.utils.data.DataLoader(dataset=TestDataset , batch_size=100, 
                                          shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, InputSize, HiddenLayersSize, classes):
        super(NeuralNet, self).__init__()
        self.InputSize = InputSize
        self.l1 = nn.Linear(InputSize, HiddenLayersSize)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(HiddenLayersSize, HiddenLayersSize)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(HiddenLayersSize, HiddenLayersSize)
        self.relu = nn.ReLU()
        self.l4 = nn.Linear(HiddenLayersSize, classes)
    
    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output= self.l2(output)
        output= self.relu(output)
        output = self.l3(output)
        output = self.relu(output)
        output = self.l4(output)
        return output

model = NeuralNet(InputSize, HiddenLayersSize, classes).to(device)

# Define Loss function and Adam optimization algorithm
LossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate)  

# The code will be used to train our model
# We are going to train our model in batches
# In one foward pass and one backward pass we will train 100 samples(ImageData)

no_Iterations = len(TrainLoader)
j = 0
while j<epochs:
    for i, (ImageData, labels) in enumerate(TrainLoader):  
    
        #flatten the image into 1D array
        ImageData = ImageData.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(ImageData)
        loss = LossFunction(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #Print after every 100 steps
        if (i+1) % 100 == 0:
            
            print (f'Epoch [{j+1}/{epochs}], Step [{i+1}/{no_Iterations}], Loss: {loss.item():.4f}')
    j+=1
    

# Since we have trained our model with 60 000 samples and our loss
# is sufficiently small and now we are ready to test our model for
# unseen data
with torch.no_grad():
    
    CorrectPrediction =0
    for ImageData, labels in TestLoader:
        
        ImageData = ImageData.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(ImageData)       
        value , pred = torch.max(outputs.data, 1)
        CorrectPrediction += (pred == labels).sum().item()
        
    accuracy = 100.0 * (CorrectPrediction / len(TestLoader.dataset))
   
    print(f'Accuracy of the network on the 10000 test ImageData: {accuracy:.2f} %')
    while True:
        UserInput = input("Please enter a filepath:\n> ")
        if(UserInput=="exit"):   
            print("Exiting....")
            break          # breaks if user enters exit
        
        Getimage = Image.open(UserInput)
        transform = transforms.ToTensor()
        ImageTensor = transform(Getimage).unsqueeze(0)
        ImageTensor  = ImageTensor.reshape(-1, 28*28).to(device)
        pred = model(ImageTensor)
        print(f"Classified: {pred.argmax()}")
