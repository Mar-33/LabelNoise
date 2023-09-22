import os
import cv2
import time
import ipdb
import yaml
import torch
import random
import argparse
import datetime
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import dataloader_phenobench ###
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassJaccardIndex

# from google.colab import drive
# drive.mount('/content/drive')

def change_instance_class(masks, old_class, new_class, factor):
  masks = masks.numpy()
  modified_masks = np.zeros_like(masks)
  for i, mask in enumerate(masks):
    mask = mask
    # Create a binary mask for the target class
    target_mask = (mask == old_class).astype(np.uint8)

    # Find connected components and label each instance
    num_labels, labeled_mask = cv2.connectedComponents(target_mask)
    random_instance = random.sample(list(np.unique(labeled_mask)[1:]),int(np.ceil(num_labels*factor)))
  
    # Replace random_instances with new class label
    instance_mask = np.isin(labeled_mask, random_instance)
    modified_masks[i] = np.where(instance_mask, new_class, mask)
  return torch.from_numpy(modified_masks)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--config",
      type=str,
      nargs="?",
      help="configfile",
  )
  opt = parser.parse_args()
  cfg = yaml.safe_load(open(opt.config))

  # Hyperparameter
  numEpochs = cfg["hyperparameter"]["numEpochs"]
  batch_size = cfg["hyperparameter"]["batch_size"]
  learning_rate = cfg["hyperparameter"]["learning_rate"]
  img_size = cfg["hyperparameter"]["img_size"]
  noise_factor = cfg["hyperparameter"]["noise_factor"]

  # num_imgs = now taking all

  # Model
  my_seed = cfg["model"]["seed"]
  encoder = cfg["model"]["encoder"]
  model_name = cfg["model"]["model_name"] + encoder + '_seed_' + str(my_seed) + '_epochs_' + str(numEpochs) + '_instance_noise_' + str(noise_factor) + '_' + str(datetime.date.today())
  num_classes = cfg["model"]["num_classes"]
  weights = cfg["model"]["weights"]

  # Prints and Save:
  print_freq = cfg["print"]["print_freq"]
  save_freq = cfg["print"]["save_freq"]

  # Paths
  img_path = cfg["path"]["img_path"]
  out_path = cfg["path"]["out_path"] + model_name + '/'
  final_model_path = os.path.join(out_path,'models/')
  checkpoint_path = os.path.join(out_path,'models/checkpoints/')
  log_path = os.path.join(out_path,'runs/')

  if not os.path.exists(out_path):
    os.makedirs(out_path)
  if not os.path.exists(final_model_path):
    os.makedirs(final_model_path)
  if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
  if not os.path.exists(log_path):
    os.makedirs(log_path)

  ######################## Fix the random seed ########################
  seed = my_seed
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  ############### Create Dataset Instance and Dataloader ###############
  transform_img = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                      transforms.ToTensor()])
  transform_mask = transforms.Compose([transforms.Resize(size=(img_size, img_size),
                                        interpolation = transforms.InterpolationMode.NEAREST), transforms.ToTensor()])

  transform = {'image': transform_img, 'mask': transform_mask}

  pheno_train_dataset = dataloader_phenobench.PhenoBenchDataset(img_path, transform=transform, split='train')
  pheno_val_dataset = dataloader_phenobench.PhenoBenchDataset(img_path, transform=transform, split='val')

  print('Number of train images: ',pheno_train_dataset.__len__())
  print('Number of val images:   ',pheno_val_dataset.__len__())

  # Create a DataLoader instance for training
  trainloader = DataLoader(pheno_train_dataset, batch_size=batch_size, shuffle=True)
  valloader = DataLoader(pheno_val_dataset, batch_size=1, shuffle=False)

  ########################## Create the model ##########################
  # Check if CUDA (GPU) is available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Selected device:", device,"\n")

  # Model
  model = smp.Unet(
      encoder_name=encoder,
      #encoder_weights=weights,
      in_channels=3,
      classes=num_classes,
      activation='softmax2d'
  )

  model.to(device)

  writer = SummaryWriter(log_path)

  ############################ Training Loop ############################

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0005)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpochs, eta_min=learning_rate/100)

  loss_max_val = float('inf')

  for epoch in range(numEpochs):
    start_time = time.time()
    print('Start Train - Epoch ', epoch)

    true_labels = []
    predicted_labels = []

    model.train()

    losses = 0
    min_train_loss = float('inf')


    for batch_idx, (img, masks) in enumerate(trainloader):
      optimizer.zero_grad()
      predictions = model(img.to(device))
      masks_squeezed = masks.squeeze(1)
      if noise_factor != 0:
        noisy_masks = change_instance_class(masks_squeezed, old_class = 1, new_class = 2, factor = noise_factor)
      else:
        noisy_masks = masks_squeezed
      loss = loss_fn(predictions,noisy_masks.long().to(device))
      loss.backward()
      optimizer.step()
      losses += loss.detach().cpu().numpy()
      if loss < min_train_loss:
        min_train_loss = loss
        # Save the model if loss has new minimum:
        #checkpoint_name_path = os.path.join(checkpoint_path, 'max_valid_model.pth')
        #torch.save(model.state_dict(), checkpoint_name_path)

      if batch_idx % print_freq == 0:
        batch_time = np.round(((time.time() - start_time)/(batch_idx+1)*(len(trainloader)-(batch_idx+1))*100)/100)
        print(f"Epoch [{epoch+1}/{numEpochs}], Batch [{batch_idx+1}/{len(trainloader)}], Time left for Epoch: {int(batch_time/(24*3600))}d {int(batch_time/3600) % 24}h {int(batch_time/60) % 60}min {int(batch_time) % 60}s, Loss: {loss.item():.4f}")
        # print('approx_time:', (np.round(((time.time() - start_time)/(batch_idx+1)*(len(trainloader-batch_idx))*100)/100)))

############################ Evaluation on Training Set ###########################

      # Convert predicted logits to class labels
      predicted_batch_labels = predictions.argmax(dim=1).cpu().numpy()
      # Convert masks to class labels
      true_batch_labels = masks.squeeze(1).cpu().numpy()
      # Collect true and predicted labels
      true_labels.extend(true_batch_labels.flatten())
      predicted_labels.extend(predicted_batch_labels.flatten())


    confusion = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_classes))
    recall = np.diag(confusion) / np.sum(confusion, axis=1)
    precision = np.diag(confusion) / np.sum(confusion, axis=0)
    accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
    iou_evaluator = MulticlassJaccardIndex(num_classes=3, average=None)
    iou = iou_evaluator(torch.tensor(predicted_labels),torch.tensor( true_labels))

    losses_mean = losses/(batch_idx+1)


    # Prints to Terminal
    print('\n-------------------- Evaluation on Training Set --------------------')
    print('Confusion Matrix:')
    print(confusion)
    training_time = np.round(((time.time() - start_time)*100)/100)
    print(f"Epoch [{epoch+1}/{numEpochs}], Training Time Epoch: {int(training_time/(24*3600))}d {int(training_time/3600) % 24}h {int(training_time/60) % 60}min {int(training_time) % 60}s, Loss(mean):{losses_mean}, Loss(min):{min_train_loss}, Accuracy: {accuracy*100:.4f}, Recall: {recall*100}, Precision: {precision*100}, IoU: {iou*100}")

    # Tensorboard: Model Performance on Training Data:
    writer.add_scalars('IoU_Training',  {'Soil':iou[0], 'Plant':iou[1], 'Weed':iou[2]}, epoch)
    writer.add_scalars('Recall_Training',  {'Soil':recall[0], 'Plant':recall[1], 'Weed':recall[2]}, epoch)
    writer.add_scalars('Precision_Training',  {'Soil':iou[0], 'Plant':iou[1], 'Weed':iou[2]}, epoch)
    writer.add_scalar('Accuracy_Training',accuracy ,epoch)
    writer.add_scalars('Confusion_Matrix_Training',  {'SS':confusion[0,0], 'SP':confusion[1,0], 'SW':confusion[2,0], 'PP':confusion[1,1], 'PS':confusion[0,1], 'PW':confusion[2,1], 'WW':confusion[2,2], 'WS':confusion[0,2], 'WP':confusion[1,2]}, epoch) # True Value --> Predicted Value
    # fig, ax = plt.subplots()
    # ConfusionMatrixDisplay(confusion).plot(ax=ax)
    # writer.add_figure("Confusion_Matrix_Training", fig, global_step=epoch)

########################### Evaluation on Validation Set ##########################
    with torch.no_grad():
      model.eval()

      val_true_labels = []
      val_predicted_labels = []


      val_losses = 0
      min_val_loss = float('inf')


      for batch_idx, (img, masks) in enumerate(valloader):
        start_val_time = time.time()

        val_predictions = model(img.to(device))
        masks_squeezed = masks.squeeze(1).long().to(device)

        val_loss = loss_fn(val_predictions,masks_squeezed)
        val_losses += val_loss.detach().cpu().numpy()
        if val_loss < min_val_loss:
          min_val_loss = val_loss

        # if batch_idx % print_freq == 0:
        #   batch_val_time = np.round(((time.time() - start_val_time)/(batch_idx+1)*(len(valloader)-(batch_idx+1))*100)/100)
        #   print(f"Epoch [{epoch+1}/{numEpochs}], Batch [{batch_idx+1}/{len(trainloader)}], Time left for Epoch: {int(batch_val_time/(24*3600))}d {int(batch_val_time/3600) % 24}h {int(batch_val_time/60) % 60}min {int(batch_val_time % 60)}s, Loss: {loss.item():.4f}")

        # Convert predicted logits to class labels
        val_predicted_batch_labels = val_predictions.argmax(dim=1).cpu().numpy()
        # Convert masks to class labels
        val_true_batch_labels = masks.squeeze(1).cpu().numpy()
        # Collect true and predicted labels
        val_true_labels.extend(val_true_batch_labels.flatten())
        val_predicted_labels.extend(val_predicted_batch_labels.flatten())

      val_losses_mean = val_losses/(batch_idx+1)

      confusion = confusion_matrix(val_true_labels, val_predicted_labels, labels=np.arange(num_classes))
      recall = np.diag(confusion) / np.sum(confusion, axis=1)
      precision = np.diag(confusion) / np.sum(confusion, axis=0)
      accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
      iou_evaluator = MulticlassJaccardIndex(num_classes=3, average=None)
      iou = iou_evaluator(torch.tensor(val_predicted_labels),torch.tensor(val_true_labels))

      # Prints to Terminal
      print('\n-------------------- Evaluation on Validation Set --------------------')
      print('Confusion Matrix:')
      print(confusion)
      validation_time = np.round(((time.time() - start_val_time)*100)/100)
      print(f"Epoch [{epoch+1}/{numEpochs}], Validation Time Epoch: {int(validation_time/(24*3600))}d {int(validation_time/3600) % 24}h {int(validation_time/60) % 60}min {int(validation_time % 60)}s, Val_Loss(mean):{val_losses_mean}, Val_Loss(min):{min_val_loss}, Accuracy: {accuracy*100:.4f}, Recall: {recall*100}, Precision: {precision*100}, IoU: {iou*100}")
      print('--------------------------------------------------------------------\n')

############################### Writing Scalars and Saving the Model ##############################

    # Losses
    writer.add_scalar('LR_Training',np.array(scheduler.get_last_lr()[0]),epoch)
    writer.add_scalars('Losses',  {'Loss_Training':losses_mean,'Loss_Evaluation':val_losses_mean}, epoch)

    # Model Performance on Validation Data:
    writer.add_scalars('IoU_Validation',  {'Soil':iou[0], 'Plant':iou[1], 'Weed':iou[2]}, epoch)
    writer.add_scalars('Recall_Validation',  {'Soil':recall[0], 'Plant':recall[1], 'Weed':recall[2]}, epoch)
    writer.add_scalars('Precision_Validation',  {'Soil':precision[0], 'Plant':precision[1], 'Weed':precision[2]}, epoch)
    writer.add_scalar('Accuracy_Validation',accuracy ,epoch)
    writer.add_scalars('Confusion_Matrix_Validation',  {'SS':confusion[0,0], 'SP':confusion[1,0], 'SW':confusion[2,0], 'PP':confusion[1,1], 'PS':confusion[0,1], 'PW':confusion[2,1], 'WW':confusion[2,2], 'WS':confusion[0,2], 'WP':confusion[1,2]}, epoch) # True Value --> Predicted Value
    # fig, ax = plt.subplots()
    # ConfusionMatrixDisplay(confusion).plot(ax=ax)
    # writer.add_figure("Confusion Matrix Validation", fig, global_step=epoch)

    # Update the Learning Rate
    scheduler.step()

    if epoch % save_freq == 0:
        # Save the model after each X-epochs
        checkpoint_name = 'checkpoint_'+'epoch_'+ str(epoch) + '.pth'
        checkpoint_name_path = os.path.join(checkpoint_path, checkpoint_name)
        torch.save(model.state_dict(), checkpoint_name_path)

    if loss_max_val > loss:
        loss_max_val = min_train_loss
        # Save the model if loss has new minimum:
        checkpoint_name_path = os.path.join(checkpoint_path, 'max_valid_model.pth')
        torch.save(model.state_dict(), checkpoint_name_path)

    # Time estimation
    epoch_time = np.round(((time.time() - start_time)*(numEpochs-(epoch+1))*100)/100)
    print(f"Time left for Training: {int(epoch_time/(24*3600))}d {int(epoch_time/3600) % 24}h {int(epoch_time/60) % 60}min {int(epoch_time % 60)}s \n")

  ### Saving Model after last Epoch
  torch.save(model.state_dict(), final_model_path + model_name + '.pth')
  writer.close()


if __name__ == "__main__":
    main()
