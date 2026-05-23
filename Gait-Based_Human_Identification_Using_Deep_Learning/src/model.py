import torch
import torch.nn as nn
import torch.nn.functional as F

class GaitModel(nn.Module):
    def __init__(self, num_classes=74):
        super(GaitModel, self).__init__()
        
        # Layer 1: Extract basic shapes
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) 
        )
        
        # Layer 2: Deeper features
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) 
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        batch_size, frames, c, h, w = x.size()
        x = x.view(-1, c, h, w) 
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        x = x.view(batch_size, frames, -1)
        x = torch.max(x, dim=1)[0]

        embedding = self.fc(x)
        return embedding

if __name__ == "__main__":
    
    model = GaitModel()
    model.eval() 
    test_input = torch.randn(1, 30, 1, 64, 64)
    
    with torch.no_grad(): 
        output = model(test_input)
        
    print(f"Model Test Successful! Output Shape: {output.shape}")