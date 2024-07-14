import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F





class MNmodelPartAM_Dueling(nn.Module):
    def __init__(self,scale,output = 128,dropout=0.25,n_actions=5,am=10,aes=10,ahes=100,n_frames=3): #action memory, action embed size, action historic embed size
        super(MNmodelPartAM_Dueling, self).__init__()
        # Input shape: [batch_size, 3, 64, 64]
        self.unet = False
        self.y_size = 10*am
        self.n_actions = n_actions
        self.am = am
        # Convolutional layers
        self.dropout = dropout
        self.output = output
        self.conv0 = nn.Conv2d(n_frames, 16*scale, kernel_size=5, stride=1,padding=1) #CHECK HOW SPATIAL DIM IS REDUCED AFTER THIS
        self.conv1 = nn.Conv2d(16*scale, 16*scale, kernel_size=3, stride=2,padding=1) #CHECK HOW SPATIAL DIM IS REDUCED AFTER THIS
        self.conv2 = nn.Conv2d(16*scale, 16*scale, kernel_size=3, stride=1,padding=1) 
        self.conv3 = nn.Conv2d(16*scale, 16*scale, kernel_size=3, stride=2,padding=1)
        self.conv4 = nn.Conv2d(16*scale, 4*scale, kernel_size=3, stride=1,padding=1)
        self.conv5 = nn.Conv2d(4*scale, 1*scale, kernel_size=3, stride=2,padding=1)
        # Fully connected layers
        # Calculate the flattened size after the convolutional layers
        self.fc1 = nn.Linear(1*scale*16*16, 1*scale*16*16)
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(1*scale*16*16, scale*8*16)
        self.dropout2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(scale*8*16, 16*16)
        self.dropout3 = nn.Dropout(self.dropout)
        self.fc4 = nn.Linear(16*16+ahes, 16*16+ahes) #FC4 concat y
        self.fc4_2 = nn.Linear(16*16+ahes, 16*16+ahes) #FC4 concat y
        self.fc4_3 = nn.Linear(16*16+ahes, 16*16+ahes) #FC4 concat y
        self.dropout4 = nn.Dropout(self.dropout)

        self.fcv1 = nn.Linear(16*16+ahes, 16*16+ahes)
        self.fcv2 = nn.Linear(16*16+ahes, 100)
        self.fcv3 = nn.Linear(100, 1)
        
        self.fc5 = nn.Linear(16*16+ahes, 16*16+ahes)
        self.dropout5 = nn.Dropout(self.dropout)
        self.fc6 = nn.Linear(16*16+ahes, self.output)
        
        self.fc1_y = nn.Linear(self.n_actions, aes)
        self.fc2_y = nn.Linear(aes*am, ahes)
        self.dropout6 = nn.Dropout(self.dropout)
        self.dropout7 = nn.Dropout(self.dropout)
        
        self.initialize_weights()
    def initialize_weights(self):
        for param in self.parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    init.xavier_uniform_(param)
                else:
                    init.constant_(param, 0)
    def forward(self, x, y, eval = False):
        # Apply convolutional layers with GeLU activation and batch normalization
        x = F.gelu(self.conv0(x))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        # Flatten the output for the fully connected layers
        x = x.flatten(1)
        

        # Apply fully connected layers with GeLU activation and dropout (except the last layer)
        x = F.gelu(self.fc1(x))
        if not eval:
            x = self.dropout1(x)
        x = F.gelu(self.fc2(x))
        if not eval:
            x = self.dropout2(x)
        x = F.gelu(self.fc3(x))
        if not eval:
            x = self.dropout3(x)


        y = F.gelu(self.fc1_y(y)) #action_mapping
        if not eval:
            y = self.dropout6(y)
        y = y.flatten(-2) #concatenate action embeds
        y = F.gelu(self.fc2_y(y))
        if not eval:
            y = self.dropout7(y)

        x = torch.cat((x,y),axis=-1)
        
        x = F.gelu(self.fc4_3(F.gelu(self.fc4_2(F.gelu(self.fc4(x))))))
        v = self.fcv3(F.gelu(self.fcv2(F.gelu(self.fcv1(x)))))
        
        x = F.gelu(self.fc5(x))
        a = self.fc6(x)
        norm_cst = a.mean()
        
        return 1,v+a-norm_cst

