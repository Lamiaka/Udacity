import argparse
import model_build as mb
import helper as h
import json
import os 

parser = argparse.ArgumentParser(description="train a neural network to classify images")
parser.add_argument("data_dir", help="the data directory")
parser.add_argument("--save_dir", help="directory where output is saved",default=None)
parser.add_argument("--arch", help="model architecture", default="vgg16")
parser.add_argument("--learning_rate", type=float, help="learning rate",default=0.0005)
parser.add_argument("--hidden_units", type=int, nargs='+',help="hidden units",default=512)
parser.add_argument("--epochs", type=int, help="number of epochs",default=5)
parser.add_argument("--dropP", type=float, help="dropout probability",default=0.2)
parser.add_argument("--gpu", action="store_true", help="activate the use of gpu",default=False)

args = parser.parse_args()

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'

print("Running on device: ", device)
print("\n")

if os.path.isdir(args.save_dir):
    print("Saving in the directory {}...".format(args.save_dir))
else:
    os.mkdir(args.save_dir)
    print("Creating the directory {} to save training output.".format(args.save_dir))
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

image_datasets, imageloader = h.dataProcessing(args.data_dir)    
model = mb.modelBuild(args.data_dir,args.save_dir,args.arch,args.learning_rate,args.hidden_units,args.epochs,args.dropP,image_datasets,cat_to_name)
model = mb.train(model,args.save_dir,imageloader,args.learning_rate,args.epochs,device,print_every = 5)
    
    