import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Function get_input_args() parses keyword arguments from the command line


def get_input_args():
    """Parses the keyword arguments from command line
    """

    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # Add architecture selection to parser
    parser.add_argument('--arch',
                        type=str,
                        help='Choose architecture \
                    (https://pytorch.org/docs/stable/torchvision/models.html)')

    # Add checkpoint directory to parser
    parser.add_argument('--save_dir',
                        type=str,
                        help='Define save directory for checkpoints')

    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Define gradient descent learning rate')
    parser.add_argument('--hidden_units',
                        type=int,
                        help='Number of hidden units for DNN classifier')
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of epochs')

    # Add GPU Option for training
    parser.add_argument('--gpu',
                        action="store_true",
                        help='Use GPU for calculations')

    # Parse args
    args = parser.parse_args()
    return args


def train_transformer(train_dir):
    """Performs training transformations on a dataset

    Args:
        train_dir (str): [Dataset training folder]

    Returns:
        [trainset]: [Torchvision dataset for the images]
    """
    # Data transformation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])
    # Load the Data
    trainset = datasets.ImageFolder(train_dir, transform=transform)
    return trainset


def test_transformer(test_dir):
    """Performs test transformations on a dataset

    Args:
        test_dir (str): [Dataset testing folder]

    Returns:
        [testset]: [Torchvision dataset for the images]
    """

    # Data transformation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])
    # Load the Data
    testset = datasets.ImageFolder(test_dir, transform=transform)
    return testset


def data_loader(data, batch_size, train=True):
    """Creates a dataloader from dataset

    Args:
        data (torchvision.dataset): [dataset to be loaded]
        batch_size (int): [batch size for dataloader]
        train (bool, optional): [Shuffle or not the dataset.
                                True for training purposes].
                                Defaults to True.

    Returns:
        [DataLoader]: [dataloader]
    """
    if train:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader


def check_gpu(gpu_arg=True):
    """Check if GPU is avalaible.

    Args:
        gpu_arg (bool, optional): [Use GPU]. Defaults to True.

    Returns:
        [torch.device]: [device]
    """

    # If gpu_arg is false then simply return the cpu device
    if gpu_arg is True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")

    # Print result
    if device == "cpu":
        print("CUDA not found on device. For more details please visit: \
        https://developer.nvidia.com/cuda-downloads.\nUsing CPU.")
    return device


def model_loader(architecture="vgg16"):
    """Download model from torchvision

    Args:
        architecture (str, optional): [model architectures for
                                      image classification].
                                      Defaults to "vgg16".

    Returns:
        [torchvision.models]: [downloaded model]
    """
    # Load Defaults if none specified
    if architecture is None:
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else:
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    return model


def initial_classifier(model, hidden_units):
    """Creates a classifier.

    Args:
        model ([torchvision.models]): [model of neural network]
        hidden_units ([int]): [Number of hidden units]

    Returns:
        [torch.nn.modules.container.Sequential]: [new classifier]
    """
    # Check that hidden layers has been input
    if hidden_units is None:
        print("Number of Hidden Layers specificed as 4096.")
        hidden_units = 4096
    else:
        hidden_units = hidden_units

    # Find number of input Layers
    input_features = model.classifier[0].in_features

    # Define Classifier
    # model.classifier[-1] = nn.Sequential(
    #                    nn.Linear(in_features=4096, out_features=102),
    #                    nn.LogSoftmax(dim=1)
    #                     )

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))]))

    return model.classifier


def validation(model, testloader, criterion, device):
    """Validates training against testloader

    Args:
        model ([torchvision.models]): [model of neural network]
        testloader ([DataLoader]): [testloader]
        criterion ([torch.nn.modules.loss]): [criteria for classifier]
        device ([torch.device]): [Device use for validation (GPU or CPU)]

    Returns:
        test_loss[torch.FloatTensor]: [test_loss]
        accuracy[torch.FloatTensor]: [accuracy]
    """
    test_loss = 0
    accuracy = 0

    for batch, (inputs, labels) in enumerate(testloader):

        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


def net_train(model, trainloader, validloader, device,
              criterion, optimizer, print_steps, steps, num_epochs):

    if num_epochs is None:
        print("Number of epochs specificed as 5.")
        num_epochs = 5
    else:
        num_epochs = num_epochs

    print("Training process initializing .....\n")
    # Train Model
    for e in range(num_epochs):
        batch_loss = 0

        for batch, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

            if batch % print_steps == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model,
                                                      validloader,
                                                      criterion,
                                                      device)

                print("Epoch: {0}/{1} | ".format(e+1, num_epochs),
                      "Training Loss: {:.4f} | ".format(
                          batch_loss/print_steps),
                      "Validation Loss: {:.4f} | ".format(
                          valid_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}".format(
                          accuracy/len(validloader)))

                batch_loss = 0
                model.train()

    return model


def validate_model(model, testloader, device):
    """Validate the above model on test data images

    Args:
        model ([torchvision.models]): [model of neural network]
        testloader ([DataLoader]): [testloader]
        device ([torch.device]): [Device use for validation (GPU or CPU)]
    """

    # Do validation on the test set
    num_correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            pred = torch.argmax(output, 1)
            total += labels.size(0)
            num_correct += (pred == labels).sum().item()

    print('Accuracy achieved by the network on test \
          images is: {} %'.format((100 * num_correct / total)))


def initial_checkpoint(model, savedir, trainset):
    """Saves the model to checkpoint.

    Args:
        model ([torchvision.models]): [model of neural network]
        savedir ([str]): [Save directory of .pth file.]
        trainset ([type]): [Torchvision dataset for the images]
    """

    if savedir is None:
        print("Model checkpoint directory is empty, model will not be saved.")
    else:
        if isdir(savedir):
            # Create `class_to_idx` attribute in model
            model.class_to_idx = trainset.class_to_idx

            # Create checkpoint dictionary
            checkpoint = {'architecture': model.name,
                          'class_to_idx': model.class_to_idx,
                          'classifier': model.classifier,
                          'state_dict': model.state_dict()}

            # Save checkpoint
            torch.save(checkpoint, 'my_checkpoint.pth')

        else:
            print("Directory not found, model will not be saved.")


# Main program function defined below
def main():

    # Get Keyword Args for Training
    in_arg = get_input_args()

    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Pass transforms in, then create trainloader
    train_data = train_transformer(train_dir)
    valid_data = test_transformer(valid_dir)
    test_data = test_transformer(test_dir)

    trainloader = data_loader(train_data, batch_size=20)
    validloader = data_loader(valid_data, batch_size=20, train=False)
    testloader = data_loader(test_data, batch_size=20, train=False)

    # Load Model
    model = model_loader(architecture=in_arg.arch)

    # Build Classifier
    model.classifier = initial_classifier(model,
                                          hidden_units=in_arg.hidden_units)

    # Check for GPU
    device = check_gpu(gpu_arg=in_arg.gpu)

    # Send model to device
    model.to(device)
    print("Using device: {}".format(device))

    # Check for learnrate args
    if in_arg.learning_rate is None:
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else:
        learning_rate = in_arg.learning_rate

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Define deep learning method
    print_steps = 40
    steps = 0

    # Train the classifier layers using backpropogation

    trained_model = net_train(model=model,
                              trainloader=trainloader,
                              validloader=validloader,
                              device=device,
                              criterion=criterion,
                              optimizer=optimizer,
                              print_steps=print_steps,
                              steps=steps,
                              num_epochs=in_arg.epochs)

    print("\nTraining process is now complete.")

    # Quickly Validate the model
    validate_model(trained_model, testloader, device)

    # Save the model
    initial_checkpoint(trained_model, in_arg.save_dir, train_data)


# Call to main function to run the program
if __name__ == '__main__':
    main()
