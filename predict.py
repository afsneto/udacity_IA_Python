import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import check_gpu
from torchvision import models


def get_input_args():
    # Define a parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # Point towards image for prediction
    parser.add_argument('--image',
                        type=str,
                        help='Point to impage file for prediction.',
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint',
                        type=str,
                        help='Point to checkpoint file as str.',
                        required=True)

    # Specify top-k
    parser.add_argument('--top_k',
                        type=int,
                        help='Choose top K matches as int.')

    # Import category names
    parser.add_argument('--category_names',
                        type=str,
                        help='Mapping from categories to real names.')

    # Add GPU Option to parser
    parser.add_argument('--gpu',
                        action="store_true",
                        help='Use GPU + Cuda for calculations')

    # Parse args
    args = parser.parse_args()

    return args


def load_checkpoint(checkpoint_path):
    """[Loads saved deep learning model from checkpoint]

    Args:
        checkpoint_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Load the saved file
    checkpoint = torch.load("my_checkpoint.pth")

    # Load Defaults if none specified
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else:
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array.

    Args:
        image_path ([str]): [image dir]

    Returns:
        [tensor]: [image in torch.tensor]
    """
    im = PIL.Image.open(image)

    width, height = im.size
    if min(width, height) > 256:
        if width < height:
            resize = [256, height / (width/256)]
        else:
            resize = [width / (height/256), 256]
    else:
        if width < height:
            resize[256, height * (256/width)]
        else:
            resize = [width * (256/height), 256]

    im.thumbnail(size=resize)

    coord_center = [width/2, height/2]
    # crop 224x224 image
    left = coord_center[0]/2 - 224/2
    right = left + 224

    top = coord_center[0]/2 - 224/2
    bottom = top + 224
    im_crop = im.crop((left, top, right, bottom))

    # convert to numpy
    np_image = np.array(im_crop)
    # maximum value 255
    np_image = (np_image)/255

    # normalize colors
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std

    # reorder dimensions
    torch_image = torch.from_numpy(np_image.transpose(2, 0, 1))

    return torch_image


def predict(image, model, device, cat_to_name, top_k=5):
    """Predict the class (or classes) of an image using a
    trained deep learning model.

    Args:
        image_path ([torch.tensor]): [Path to image]
        model ([torchvision.models]): [model of neural network]
        device ([torch.device]): [Device use for validation (GPU or CPU)]
        cat_to_name ([type]): [description]
        top_k ([int]): [he top K classes to be calculated]

    Returns:
        top_probs ([list]): [probabilities of the K classes]
        top_labels ([list]): [labels of the K classes]

    """
    if top_k is None:
        top_k = 5
    else:
        top_k = top_k

    # Set model to evaluate
    model.eval()

    torch_image = process_image(image)
#     print(torch_image.shape)
    torch_image = torch.unsqueeze(torch_image, 0)
    torch_image = torch_image.to(device, dtype=torch.float)
#     print(torch_image.shape)

    model = model.to(device)

    logps = model(torch_image)
    output = torch.exp(logps)
    # top_k results
    top_probs, top_labels = output.topk(top_k, dim=1)
    top_probs = top_probs.tolist()[0]
    top_labels = top_labels.tolist()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    """
    Converts two lists into a dictionary to print on screen
    """

    for i, j in enumerate(zip(flowers, probs)):
        print("Rank {}:".format(i+1),
              "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))


# Main program function defined below
def main():

    # Get Keyword Args for Prediction
    args = get_input_args()

    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)

    # Process Image
    # image_tensor = process_image(args.image)   ######
    image_tensor = args.image

    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu)

    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensor, model,
                                                 device, cat_to_name,
                                                 args.top_k)

    # Print out probabilities
    print_probability(top_flowers, top_probs)


# Call to main function to run the program
if __name__ == '__main__':
    main()
