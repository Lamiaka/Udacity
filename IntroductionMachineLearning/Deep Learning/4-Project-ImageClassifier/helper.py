import torch
from torchvision import datasets, transforms, models
import os
from PIL import Image
import numpy as np

def getClassCount(data_dir):
    ''' Count number of classes based on number of folders in data directory.
    
        Arguments:
        ---------
        data_dir: directory name where data is stored in folders, one folder for each class
        
        Output: 
        -------
        number of classes
    '''
    
    path, dirs, files = next(os.walk(os.path.join(data_dir,'train')))
    class_count = len(dirs)
    
    return class_count

def dataLoad(data_dir):
 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    dirs = [train_dir,valid_dir,test_dir]
    sets = ['train','valid','test']

    return sets, dirs

def dataProcessing(data_dir,batch_size =32,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    
    data_transforms = {'train': (transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])),
                                               
                        'test': (transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)])),
                   
                       'valid': (transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)]))}


    image_datasets = dict()
    imageloader = dict()
    sets, dirs = dataLoad(data_dir)

    for s, d in zip(sets, dirs):
        image_datasets[s] = datasets.ImageFolder(d,transform = data_transforms[s])
        if s == 'train':
            imageloader[s] = torch.utils.data.DataLoader(image_datasets[s], batch_size = batch_size,shuffle = True)
        else:
            imageloader[s] = torch.utils.data.DataLoader(image_datasets[s], batch_size = batch_size)

    return image_datasets, imageloader


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor.
    '''
    resize = 256
    cropsize = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    size = image.size
    aspect_ratio = size[0]/size[1]
    i_min = np.argmin(size)
        
    # resizing the image
    if size[0] == size[1]:
        w, h = resize, resize
            
    elif i_min == 1:
        w, h = aspect_ratio * resize, resize
            
    else:
        w, h = resize, aspect_ratio* resize
            
    image.thumbnail((w,h))
        
    # cropping the center
    pix_w = np.floor((w-cropsize)/2)
    pix_h = np.floor((h-cropsize)/2)
    crop_pix = [pix_w,pix_h,w-pix_w,h-pix_h]
    image_c = image.crop(crop_pix)
    image_c2 = image.crop((0,0,cropsize,cropsize))

    np_image = np.array(image_c2)
    img_max = np_image.max()
    np_image_norm = np_image/img_max
    np_image_stnd = (np_image_norm-mean)/std
        
    image = np_image_stnd.transpose(2,0,1)
    image = torch.from_numpy(image).float()
    return image

def imagePrep(image_path):
    image = Image.open(image_path)
    image = process_image(image)
    
    return image

def predict(image, model, topk=5, device = 'cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''     
    
    image.unsqueeze_(0)
           
    model.to(device)
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model.forward(image)

    ps = torch.exp(output)
    
    probs, idx = ps.topk(topk, dim=1)
    idx = idx.to('cpu').numpy()[0]
    class_ind = [model.classifier.idx_to_class[c] for c in idx]
    classes = [model.classifier.class_to_name[c] for c in class_ind]
    probs = probs.to('cpu').numpy()[0]
    
    return probs, classes