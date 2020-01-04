import torch
import os
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from collections import OrderedDict
import helper as h



# Building the network
class Classifier(nn.Module):
    
    def __init__(self,input_size,output_size,hidden_layers,drop_p):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop: float, dropout probability
        
        '''
        super().__init__()
        
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # Output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        # Dropout
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self,x):
        ''' forward method to propagate the model input through the network, returns the output logits 
        
            Arguments
            ---------
            x: input object to the model
        '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
         
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    

def modelBuild(data_dir,save_dir,arch,learning_rate,hidden_units,epochs,dropoutP,image_datasets,cat_to_name):
    
    model = getattr(models, arch)(pretrained=True)
    input_size = model.classifier[0].in_features
    output_size = h.getClassCount(data_dir)

    for param in model.parameters():
        param.requires_grad = False

    classifier = Classifier(input_size,output_size,hidden_units,dropoutP)
    model.classifier = classifier
 
    if save_dir !=None: 
        class_to_idx = image_datasets['train'].class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        idx_to_name = {i: cat_to_name[idx_to_class[i]] for i in idx_to_class.keys()}
        
        checkpoint = {'input_size': input_size,
                      'output_size': output_size,
                      'hidden_units': hidden_units,
                      'drop': dropoutP,
                      'epochs': epochs,
                      'learning_rate': learning_rate,
                      'arch': arch,
                      'class_to_idx': class_to_idx,
                      'idx_to_class': idx_to_class,
                      'class_to_name': cat_to_name,
                      'idx_to_name': idx_to_name,
                      'state_dict': model.classifier.state_dict()}

        torch.save(checkpoint, os.path.join(save_dir,'checkpoint_init.pth'))
    return model

               
#training the model              
def train(model,save_dir,imageloader,learning_rate,epochs,device,print_every = 5):
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate) 
    criterion = nn.NLLLoss()
    names = ['steps']                            
    train_losses, valid_losses, valid_accuracies = ['train losses'], ['valid losses'], ['valid accuracies']
    running_loss = 0
    steps = 0
    if save_dir !=None:  
        checkpoint = torch.load(os.path.join(save_dir,'checkpoint_init.pth'))
        checkpoint['optimizer'] = optimizer
        training_performance_path = os.path.join(save_dir,'training_performance.csv')
        checkpoint_path = os.path.join(save_dir,'checkpoint_train.pth')
    
    model.to(device)
               
    for epoch in range(epochs): 
    
        for images, labels in imageloader['train']:
            steps += 1  
        
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps%print_every == 0:
                with torch.no_grad():
                    model.eval()
                    valid_loss = 0
                    valid_accuracy = 0
            
                    for images, labels in imageloader['valid']:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        loss = criterion(logps,labels)
                        valid_loss += loss.item()
                
                        #calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
                print(15*'--')
                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Steps: {}".format(steps),
                  "\nTraining Loss: {:.3f}.. ".format(running_loss/len(imageloader['train'])),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(imageloader['valid'])),
                  "Valid Accuracy: {:.3%}".format(valid_accuracy/len(imageloader['valid'])))
                print(15*'--'+'\n')
                            
                if save_dir != None:
                    # Saving performance metrics
                    names.append(steps)
                    train_losses.append(running_loss/len(imageloader['train']))
                    valid_losses.append(valid_loss/len(imageloader['valid']))
                    valid_accuracies.append(valid_accuracy/len(imageloader['valid']))
                    np.savetxt(training_performance_path, [p for p in zip(names, train_losses, valid_losses, valid_accuracies)], delimiter=',',fmt='%s')
                  
                    # Saving model checkpoint
                    checkpoint['state_dict'] = model.classifier.state_dict()
                    torch.save(checkpoint, checkpoint_path)
                
                running_loss = 0 
                model.train()
                
    return model

def loadCheckpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    # Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = Classifier(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_units'],
                         checkpoint['drop'])
    model.classifier = classifier
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.classifier.optimizer = checkpoint['optimizer']
    model.classifier.epochs = checkpoint['epochs']
    model.classifier.learning_rate = checkpoint['learning_rate']
    model.classifier.class_to_idx = checkpoint['class_to_idx']
    model.classifier.idx_to_class = checkpoint['idx_to_class']
    model.classifier.class_to_name = checkpoint['class_to_name']
    
    return model