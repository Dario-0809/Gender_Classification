from library import *

# Create network
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        # Hidden Layer 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=32)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)
       
        # Hidden Layer 2
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(num_features=64)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        
        # Hidden Layer 3
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=128)
        self.relu3=nn.ReLU()
        self.pool3=nn.MaxPool2d(kernel_size=2)

        # Hidden Layer 4
        self.conv4=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn4=nn.BatchNorm2d(num_features=256)
        self.relu4=nn.ReLU()
        self.pool4=nn.MaxPool2d(kernel_size=2)

        # Hidden Layer 5
        self.conv5=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn5=nn.BatchNorm2d(num_features=512)
        self.relu5=nn.ReLU()
        self.pool5=nn.MaxPool2d(kernel_size=2)

        #fully connected
        self.fc1=nn.Linear(in_features=512*7*7,out_features=2048)
        self.relu6=nn.ReLU()
        self.fc2=nn.Linear(in_features=2048, out_features=1024)
        self.relu7=nn.ReLU()
        self.fc3=nn.Linear(in_features=1024, out_features=512)
        self.relu8=nn.ReLU()
        self.fc4=nn.Linear(in_features=512, out_features=128)
        self.relu9=nn.ReLU()
        self.fc5=nn.Linear(in_features=128, out_features=32)
        self.relu10=nn.ReLU()
        self.fc6=nn.Linear(in_features=32, out_features=2)

        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)  
        output=self.pool1(output)
   
        output=self.conv2(output) 
        output=self.bn2(output)
        output=self.relu2(output)
        output=self.pool2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        output=self.pool3(output)

        output=self.conv4(output)
        output=self.bn4(output)
        output=self.relu4(output)
        output=self.pool4(output)
        
        output=self.conv5(output)
        output=self.bn5(output)
        output=self.relu5(output)
        output=self.pool5(output)

        output = output.view(-1, 512*7*7)

        output=self.fc1(output)
        output=self.relu6(output)
        output=self.fc2(output)
        output=self.relu7(output)
        output=self.fc3(output)
        output=self.relu8(output)
        output=self.fc4(output)
        output=self.relu9(output)
        output=self.fc5(output)
        output=self.relu10(output)
        output=self.fc6(output)

        return output
