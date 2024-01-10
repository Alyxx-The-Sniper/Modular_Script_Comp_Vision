import torch
import torchvision
import argparse
import model_builder
#######################################################################################

# PARSER

# Creating a parser
parser = argparse.ArgumentParser()

# parser for image and model
# Get an image path
parser.add_argument("--image",
                    help="target image filepath to predict on")

# Get a model path
parser.add_argument("--model_path",
                    default="models/05_going_modular_script_mode_tinyvgg_model.pth",
                    type=str,
                    help="target model to use for prediction filepath")

args = parser.parse_args()

#######################################################################################
# to gert the attributes `classnames`
from pathlib import Path
data_path = Path("data/")
train_dir = image_path / "train"
data_transform_train_forclass = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
train_data_forclass = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform_train, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)
# Setup class names (HARD CODED, we can create an extension code to retrieve class_names)
# class_names = ['cup_cakes', 'donuts', 'french_fries', 'ice_cream']
class_names = train_data_forclass.classnames
print(class_names)




# Setup class names (HARD CODED, we can create an extension code to retrieve class_names)
class_names = ['cup_cakes', 'donuts', 'french_fries', 'ice_cream']

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the image path
IMG_PATH = args.image
print(f"[INFO] Predicting on {IMG_PATH}")

# Function to load in the model
def load_model(filepath=args.model_path):
  # Need to use same hyperparameters as saved model
  model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=10,
                                output_shape=4).to(device)

  print(f"[INFO] Loading in model from: {filepath}")
  # Load in the saved model state dictionary from file
  model.load_state_dict(torch.load(filepath))

  return model

# Function to load in model + predict on select image
def predict_on_image(image_path=IMG_PATH, filepath=args.model_path):
  # Load the model
  model = load_model(filepath)

  # Load in the image and turn it into torch.float32 (same type as model)
  image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

  # Preprocess the image to get it between 0 and 1
  image = image / 255.

  # Resize the image to be the same size as the model
  transform = torchvision.transforms.Resize(size=(64, 64), antialias=True)
  image = transform(image)

  # Predict on image
  model.eval()
  with torch.inference_mode():
    # Put image to target device
    image = image.to(device)

    # Get pred logits
    pred_logits = model(image.unsqueeze(dim=0)) # make sure image has batch dimension (shape: [batch_size, height, width, color_channels])

    # Get pred probs
    pred_prob = torch.softmax(pred_logits, dim=1)

    # Get pred labels
    pred_label = torch.argmax(pred_prob, dim=1)
    pred_label_class = class_names[pred_label]

  print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_prob.max():.3f}")

if __name__ == "__main__":
  predict_on_image()