import argparse
import helper as h 
import model_build as mb
import json
from tabulate import tabulate


parser = argparse.ArgumentParser(description = "predict the class of an image and giving its probability")
parser.add_argument("path", help="path to the image to classify")
parser.add_argument("checkpoint", help="path to checkpoint to build the classifier")
parser.add_argument("--top_k", type=int, choices=range(1, 20), help="top K most likely classes",default=5)
parser.add_argument("--category_names", help="path to json file that maps values to category names", default=None)
parser.add_argument("--gpu", action="store_true", help="activate the use of gpu",default=False)

args = parser.parse_args()

if args.gpu:
    device = 'cuda'
    dp = 'gpu'
else:
    device = 'cpu'
    dp = 'cpu'

print("Running on ", dp)
print("\n")

if args.category_names != None:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

model = mb.loadCheckpoint(args.checkpoint)
image = h.imagePrep(args.path)
probs, classes = h.predict(image,model,args.top_k,device)

print("The image is predicted to be in class {c} with probability {p:.2%}.".format(c = classes[0], p=probs[0]))
      
if args.top_k > 1:
    print("\nSummary:\n")
    print("%(c)-20s %(p)-s" % {"c": 'Class', "p": 'Probability'})
    print("%(c)-20s %(p)-s" % {"c": '----------', "p": '----------'})
    for c,p in zip(classes,probs):    
        print("{c:20s} {p:.2%}".format(c=c,p=p))
                