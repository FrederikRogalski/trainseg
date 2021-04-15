#%tensorflow_version 1.x
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm, trange
import cv2

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

"""
Author: Manpreet Singh Minhas
Contact: msminhas at uwaterloo ca
"""
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')


parser.add_argument('--epochs', default=50, help='Number of epochs to train.', type=int)
parser.add_argument('--batch', default=20, help='Number of training examples per batch.', type=int)
parser.add_argument('--weights', default=None, help='Path to pretrained weights.')
parser.add_argument('--model', default=None, help='Path to saved model.')
parser.add_argument('--toTflite', action='store_true')
parser.add_argument('--data', default="../data/data_dir")
parser.add_argument('--wandb', action='store_true')

args = parser.parse_args()

epochs = args.epochs
batch = args.batch
weights = args.weights
model_path = args.model
toTflite = args.toTflite
data_path = args.data
wandb_flag = args.wandb

if wandb_flag:
  import wandb
  wandb.init(project="trainseg")

class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale",
                 flip: bool = True) -> None:
                 
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.

        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode
        self.flip = flip

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        if self.flip:
            return len(self.image_names)*2
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        flip_this=False
        if self.flip:
            flip_this = index >= (self.__len__()//2)
        if flip_this:
            index = index - (self.__len__()//2)
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path,
                                                        "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            sample = {"image": image, "mask": mask}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
            sample["image"] = cv2.resize(np.array(sample["image"]), dsize=(224,224))
            sample["mask"] = cv2.resize(np.array(sample["mask"]), dsize=(224,224))
            if flip_this == True:
                sample['image'] = tf.image.flip_left_right(sample['image'])
                sample['mask'] = tf.image.flip_left_right(np.expand_dims(sample['mask'], axis=2))[:,:,0]
            return sample

def get_dataloader_single_folder_tf(data_dir: str,
                                 image_folder: str = 'Images',
                                 mask_folder: str = 'Masks',
                                 fraction: float = 0.2,
                                 batch_size: int = 4):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.

    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = None #transforms.Compose([transforms.ToTensor()])
    image_datasets = {
        x: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }

    def gen_train():
      ds = iter(image_datasets['Train'])
      for data in ds:
        yield data
    def gen_test():
      ds = iter(image_datasets['Test'])
      for data in ds:
        yield data
    output_shape={'image':tf.float32,'mask':tf.float32}
    dataloaders = {
        'Train': tf.data.Dataset.from_generator(gen_train, output_types=output_shape).shuffle(buffer_size=50).batch(batch_size),
        'Test': tf.data.Dataset.from_generator(gen_test, output_types=output_shape).shuffle(buffer_size=50).batch(batch_size)
    }
    #dataloaders = {
    #    x: DataLoader(image_datasets[x],
    #                  batch_size=batch_size,
    #                  shuffle=True,
    #                  num_workers=2)
    #    for x in ['Train', 'Test']
    #}
    return dataloaders


OUTPUT_CHANNELS = 1

base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

from tensorflow_examples.models.pix2pix import pix2pix

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[224, 224, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

dl = get_dataloader_single_folder_tf(data_dir=data_path, batch_size=batch, fraction=0.05)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

# Iterate over epochs.
dl_len = None
current_best = 10e20

if model_path!=None:
  model = tf.keras.models.load_model(model_path)
if weights!=None:
  model.load_weights('trainseg_weights_best')

for epoch in range(epochs):

  # Iterate over the batches of the dataset.
  
  with tqdm(enumerate(dl['Train']), total=dl_len, desc=f"Epoch {epoch}/{epochs}") as t:
    for step, data in t:
      in0 = data['image']
      mask = data['mask']
      with tf.GradientTape() as tape:
        reconstructed = model(in0)
        # Compute reconstruction loss
        loss = mse_loss_fn(mask, reconstructed)
        loss += sum(model.losses)  # Add KLD regularization loss

      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      loss_metric(loss)
      t.set_postfix(loss=f"{loss_metric.result().numpy():.4f}")
    if wandb_flag:
      wandb.log({'epoch': epoch, 'loss_train': loss.numpy()})
    loss=0
    for data in dl['Test']:
      in0 = data['image']
      mask = data['mask']
      with tf.GradientTape() as tape:
        reconstructed = model(in0)
        # Compute reconstruction loss
        loss += mse_loss_fn(mask, reconstructed)
        loss += sum(model.losses)  # Add KLD regularization loss
    print("Test loss:", loss.numpy(), "Current best:", current_best)
    if wandb_flag:
      wandb.log({'epoch': epoch, 'loss_test': loss.numpy()})
    if loss.numpy() < current_best:
      print("Saving new best model!")
      model.save_weights('trainseg_weights_best')
      current_best = loss.numpy()
    if epoch %20:
      model.save_weights('trainseg_weights_last')
  dl_len=step

print("\nSaving Model")
model.save('trainseg_last')
if toTflite:
  def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 224, 224, 3)*255
      yield [data.astype(np.float32)]
  
  print("\nConverting to tflite")
  converter = tf.lite.TFLiteConverter.from_saved_model("trainseg_last")
  converter.experimental_new_converter = False
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset
  tflite_quant_model = converter.convert()
  
  with open('trainseg_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)
