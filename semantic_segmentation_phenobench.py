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

def instance_noise(masks, old_class, new_class, factor):
  if factor == 0:
      return masks
  else:
    masks = masks.numpy()
    modified_masks = np.zeros_like(masks)
    for i, mask in enumerate(masks):
      if old_class in mask:
        # Create a binary mask for the target class
        target_mask = (mask == old_class).astype(np.uint8)

        # Find connected components and label each instance
        num_labels, labeled_mask = cv2.connectedComponents(target_mask)

        random_instance = np.random.choice(np.unique(labeled_mask)[1:],int(np.ceil(num_labels*factor)))

        # random_instance = random.sample(list(np.unique(labeled_mask)[1:]),int(np.ceil(num_labels*factor)))

        # Replace random_instances with new class label
        instance_mask = np.isin(labeled_mask, random_instance)
        modified_masks[i] = np.where(instance_mask, new_class, mask)
      else: modified_masks[i] = mask
    return torch.from_numpy(modified_masks)

def random_noise(masks, num_classes, device, factor):
  if factor == 0:
    return masks
  else:
    noise_level = torch.randint(0, 101, (masks.shape)).to(device)
    noise_level[noise_level > factor*100] = 0
    noise_level[noise_level != 0] = 1

    noise = torch.randint(1,num_classes,(masks.shape)).to(device) * noise_level
    noisy_masks = masks.to(device) + noise.to(device)
    noisy_masks[noisy_masks>(num_classes-1)] -= num_classes

    return noisy_masks
  
def erosion(masks, num_classes, device, iter, kernel_size, er_dil_factor):
  if iter == 0:
    return masks
  else:
    erosed_masks = np.zeros((masks.shape))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    samples = np.random.choice(np.arange(0, masks.shape[0]), int(np.ceil(masks.shape[0]*er_dil_factor)), replace=False)
    for idx in range(masks.shape[0]):
      if idx in samples:
        for sem_class in np.unique(masks[idx].cpu().numpy())[1:]:
          mask = (masks[idx] == sem_class)*255
          mask = (mask.cpu().numpy()).astype(np.uint8)
          erosed = cv2.erode(mask, kernel, iterations=iter)
          erosed_masks[idx][erosed != 0] = sem_class
      else:
        erosed_masks[idx] = masks[idx].cpu().numpy()
    return torch.tensor(erosed_masks).to(device)

def dilation(masks, num_classes, device, iter, kernel_size, er_dil_factor):
  if iter == 0:
    return masks
  else:
    dilated_masks = np.zeros((masks.shape))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    samples = np.random.choice(np.arange(0, masks.shape[0]), int(np.ceil(masks.shape[0]*er_dil_factor)), replace=False)
    for idx in range(masks.shape[0]):
      if idx in samples:
        for sem_class in np.unique(masks[idx].cpu().numpy())[1:]:
          mask = (masks[idx] == sem_class)*255
          mask = (mask.cpu().numpy()).astype(np.uint8)
          dilated = cv2.dilate(mask, kernel, iterations=iter)
          dilated_masks[idx][dilated != 0] = sem_class
      else:
        dilated_masks[idx] = masks[idx].cpu().numpy()
    return torch.tensor(dilated_masks).to(device)

def generate_random_oval(image_shape, max_radius):
  h, w = image_shape
  min_radius = np.ceil(max_radius/10)
  # Generate random ellipse parameters
  center = np.random.randint(max_radius, h - max_radius, size=(2,))
  axis1 = np.random.randint(min_radius, max_radius)
  axis2 = np.random.randint(min_radius, max_radius/2)
  axes = (axis1,axis2)
  angle = np.random.randint(0, 180)
  # Create Mask
  mask = np.zeros((h, w), dtype=np.uint8)
  cv2.ellipse(mask, tuple(center), tuple(axes), angle, 0, 360, 1, -1)
  return mask

def add_instances(masks, images, min_instances, max_instances, new_class, max_radius, green_factor):
  if max_instances == 0:
    return masks, images
  else:
    num_instances = np.random.randint(min_instances,max_instances,masks.shape[0])
    # get mean and std rgb values for weed of the batch:
    # r_std_new, r_mean_new = torch.std_mean(images[:,0,:,:][masks==2])
    g_std_new, g_mean_new = torch.std_mean(images[:,1,:,:][masks==2])
    # b_std_new, b_mean_new = torch.std_mean(images[:,2,:,:][masks==2])

    for mask in range(masks.shape[0]):
      random_shape = torch.zeros(masks.shape[1:])
      for i in range(num_instances[mask]):
        random_shape += generate_random_oval(masks[mask].shape, max_radius = max_radius)
        # Füge die zufällige Form zur originalen Maske hinzu
        masks[mask][random_shape != 0] = new_class # [(random_shape != 0) & (masks[mask] == 0)]
      
      # images[mask][0,:,:][(random_shape != 0)] = (images[mask][0,:,:][(random_shape != 0)] - torch.mean(images[mask][0,:,:][(random_shape != 0)])) * r_std_new/2/torch.std(images[mask][0,:,:][(random_shape != 0)]) + r_mean_new
      images[mask][1,:,:][(random_shape != 0)] = (images[mask][1,:,:][(random_shape != 0)] - torch.mean(images[mask][1,:,:][(random_shape != 0)])) * g_std_new/torch.std(images[mask][1,:,:][(random_shape != 0)]) + g_mean_new
      # images[mask][2,:,:][(random_shape != 0)] = (images[mask][2,:,:][(random_shape != 0)] - torch.mean(images[mask][2,:,:][(random_shape != 0)])) * b_std_new/2/torch.std(images[mask][2,:,:][(random_shape != 0)]) + b_mean_new

      
      # images[mask][1,:,:][random_shape != 0] = (images[mask][1,:,:][random_shape != 0]*(255-green_factor)+green_factor)/255
      # images[mask][2,:,:][random_shape != 0] = (images[mask][2,:,:][random_shape != 0]/255*(255-green_factor/2))


      # images[mask][0,:,:][random_shape != 0] = (images[mask][0,:,:][random_shape != 0]/255*(255-green_factor))
      # images[mask][1,:,:][random_shape != 0] = (images[mask][1,:,:][random_shape != 0]*(255-green_factor)+green_factor)/255
      # images[mask][2,:,:][random_shape != 0] = (images[mask][2,:,:][random_shape != 0]/255*(255-green_factor/2))
    return masks, images
  
def cut_instance(masks, class2cut, cut_instance_factor, cut_factor, device):
  if cut_instance_factor == 0:
    return masks
  else:
    # masks = masks.cpu().numpy()
    # modified_masks = masks.copy()
    for i, mask in enumerate(masks.cpu().numpy()):
      # Create a binary mask for the target class
      target_mask = (mask == class2cut).astype(np.uint8)

      # Find connected components and label each instance
      num_labels, labeled_mask = cv2.connectedComponents(target_mask)
      if num_labels > 1:
        random_instance = random.sample(list(np.unique(labeled_mask)[1:]),int(np.ceil(num_labels*cut_instance_factor)))

        # Replace random_instances with new class label
        for rand_inst in random_instance:
          instance_mask = np.isin(labeled_mask, rand_inst)

          y, x = np.where(instance_mask == 1)
          min_y_index = y[np.argmin(y)]
          max_y_index = y[np.argmax(y)]
          min_x_index = x[np.argmin(x)]
          max_x_index = x[np.argmax(x)]

          height = int((max_y_index - min_y_index) * cut_factor)
          width = int((max_x_index - min_x_index) * cut_factor)

          my_sample = random.randint(0, 3)
          if my_sample == 0:
            instance_mask[min_y_index+height:, :] = False
          elif my_sample == 1:
            instance_mask[:,min_x_index+width:] = False
          elif my_sample == 2:
            instance_mask[:max_y_index-height, :] = False
          elif my_sample == 3:
            instance_mask[:,:max_x_index-width] = False

          masks[i] = masks[i] * (~instance_mask).astype(int)
    return masks #torch.tensor(masks).to(device)

# def leaf_noise(masks, leafs, new_class, leaf_noise_factor, device):
#   if leaf_noise_factor == 0:
#     return masks
#   else:
#     # ipdb.set_trace()
#     leaf = torch.tensor(leafs).long().to(device)

#     for i, mask in enumerate(masks):
#       leaf_ids = torch.unique(leaf[i]).cpu().detach().numpy()
#       # ipdb.set_trace()
#       if leaf_ids.size > 1:
#         random_instance = np.random.choice(leaf_ids[1:],int(np.ceil(len(leaf_ids)*leaf_noise_factor)))
#         masks[i][ torch.isin(leaf[i],torch.tensor(random_instance).to(device))] = new_class
#     return masks

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
  # Instance Noise
  in_factor = cfg["hyperparameter"]["instance_noise_factor"]
  in_old_class = cfg["hyperparameter"]["instance_noise_old_class"]
  in_new_class = cfg["hyperparameter"]["instance_noise_new_class"]
  # Random Noise
  rn_factor = cfg["hyperparameter"]['random_noise_factor']
  # Dilation and Erosion
  er_dil_factor = cfg["hyperparameter"]['er_dil_factor']
  dilation_first = cfg["hyperparameter"]['dilation_first']
  erosion_iter = cfg["hyperparameter"]['erosion_iter']
  dilation_iter = cfg["hyperparameter"]['dilation_iter']
  kernel_size_di_er = cfg["hyperparameter"]['kernel_size_di_er']
  # Add Instances
  ai_class = cfg["hyperparameter"]['add_instances_class']
  ai_min = cfg["hyperparameter"]['add_instances_min']
  ai_max = cfg["hyperparameter"]['add_instances_max']
  ai_rad = cfg["hyperparameter"]['add_instances_radius']
  ai_green = cfg["hyperparameter"]['add_instances_green']
  # Cut Instances
  cut_class = cfg["hyperparameter"]['cut_class']
  cut_instance_factor = cfg["hyperparameter"]['cut_instance_factor']
  cut_factor = cfg["hyperparameter"]['cut_factor']
  # Leaf Noise
  leaf_noise_factor = cfg["hyperparameter"]['leaf_noise_factor']
  leaf_class = cfg['hyperparameter']['leaf_class']
  # num_imgs = now taking all

  # Model
  my_seed = cfg["model"]["seed"]
  encoder = cfg["model"]["encoder"]
  my_dataset = cfg["model"]["model_name"]
  model_name = my_dataset + encoder + '_seed_' + str(my_seed) + '_in_' + str(int(in_factor*100)) + '_rn_' + str(int(rn_factor*100)) + '_di_' + str(int(dilation_iter)) + '_er_' + str(int(erosion_iter)) + '_k' + str(kernel_size_di_er) + '_def_' + str(int(er_dil_factor*100)) + '_difi_' + str(int(dilation_first)) + '_ai_' + str(int(ai_min)) + '_' + str(int(ai_max)) + '_' + str(int(ai_rad)) + '_' + str(int(ai_green)) + '_cc_' + str(int(cut_class)) +'_ci_' + str(int(cut_instance_factor*100))  + '_cf_' + str(int(cut_factor*100)) + '_lfn_once_' + str(int(leaf_noise_factor*100)) + '_' + str(datetime.date.today())
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

  ################## Check if CUDA (GPU) is available ##################

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Selected device:", device,"\n")

  ############### Create Dataset Instance and Dataloader ###############
  ## Import Dataloader for dataset:
  if my_dataset == 'phenobench_':
    import dataloader_phenobench as my_dataloader
  if my_dataset == 'cropandweed_': 
    import dataloader_cropandweed as my_dataloader

  transform_img = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                      transforms.ToTensor()])
  transform_mask = transforms.Compose([transforms.Resize(size=(img_size, img_size),
                                        interpolation = transforms.InterpolationMode.NEAREST), transforms.PILToTensor()])

  transform = {'image': transform_img, 'mask': transform_mask}


  if leaf_noise_factor > 0:
    train_dataset = my_dataloader.Dataset(img_path, transform=transform, split='train', leaf_instances=True, leaf_class = leaf_class, leaf_noise_factor = leaf_noise_factor, device = device)
    # for idx in range(train_dataset.__len__()):
    #   modified_semantics = leaf_noise(train_dataset.data[idx]['semantics'], train_dataset.data[idx]['leaf_instances'], new_class = leaf_class , leaf_noise_factor = leaf_noise_factor, device = device)
    #   ipdb.set_trace()
    #   train_dataset.data[idx]['semantics'] = modified_semantics
    #   if idx < 4:
    #     plot_image = Image.fromarray((train_dataset.data[idx]['semantics']/(num_classes-1)*255).astype('uint8'))
    #     plt.imshow(plot_image, interpolation='nearest', cmap = 'viridis')
    #     plt.colorbar()
    #     plt.show()
  else:
    train_dataset = my_dataloader.Dataset(img_path, transform=transform, split='train', leaf_instances=False, leaf_class = None, leaf_noise_factor = None, device = None)


  val_dataset = my_dataloader.Dataset(img_path, transform=transform, split='val')
  

  print('Number of train images: ',train_dataset.__len__())
  print('Number of val images:   ',val_dataset.__len__())

  # Create a DataLoader instance for training
  trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

  ########################## Create the model ##########################


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
      # if epoch == 1:
      #   plot_image = Image.fromarray(np.transpose((img[0].numpy()*255).astype('uint8'),(1,2,0)))
      #   plt.imshow(plot_image, interpolation='nearest', cmap = 'viridis')
      #   plt.colorbar()
      #   plt.show()
      optimizer.zero_grad()
      noisy_masks = masks.squeeze(1).clone()
      images = img.to(device)
      
      # for each in range(batch_size):

      #   plot_image = Image.fromarray((masks.squeeze(1)[each].numpy()/(num_classes-1)*255).astype('uint8'))
      #   plt.imshow(plot_image, interpolation='nearest', cmap = 'viridis')
      #   plt.colorbar()
      #   plt.show()

      # Label Augmentations:
      noisy_masks = instance_noise(noisy_masks, old_class = in_old_class, new_class = in_new_class, factor = in_factor) # Instance Noise (Plant2Weed)
      noisy_masks = cut_instance(noisy_masks, class2cut = cut_class, cut_instance_factor = cut_instance_factor, cut_factor = cut_factor, device=device)
      noisy_masks, images = add_instances(noisy_masks, images, min_instances = ai_min, max_instances = ai_max, new_class = ai_class, max_radius = ai_rad, green_factor = ai_green)
      # noisy_masks = leaf_noise(noisy_masks, leafs, new_class = leaf_class , leaf_noise_factor = leaf_noise_factor, device = device)
      if dilation_first == True:
        noisy_masks = dilation(noisy_masks, num_classes, device, iter = dilation_iter, kernel_size = kernel_size_di_er, er_dil_factor=er_dil_factor) # Dilation
        noisy_masks = erosion(noisy_masks, num_classes, device, iter = erosion_iter, kernel_size = kernel_size_di_er, er_dil_factor=er_dil_factor) # Erosion
      if dilation_first == False:
        noisy_masks = erosion(noisy_masks, num_classes, device, iter = erosion_iter, kernel_size = kernel_size_di_er, er_dil_factor=er_dil_factor) # Erosion
        noisy_masks = dilation(noisy_masks, num_classes, device, iter = dilation_iter, kernel_size = kernel_size_di_er, er_dil_factor = er_dil_factor) # Dilation
      noisy_masks = random_noise(noisy_masks, num_classes,device, rn_factor) # Random Noise

      # # Anzahl der Spalten und Zeilen im Subplot
      # num_rows = 2  # Anzahl der Zeilen
      # num_cols = 4  # Anzahl der Spalten

      # # Erstellen des Subplots
      # fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))

      # # Iteration durch die Bilder und Anzeige in den Subplots
    
      # for each in range(masks.shape[0]):
      #     ## Images
      #     ax = axes[0, each]
      #     plot_mask = Image.fromarray((noisy_masks[each].numpy() / (num_classes - 1) * 255).astype('uint8'))
      #     ax.imshow(plot_mask, interpolation='nearest', cmap='viridis', vmin = 0, vmax = 255)
      #     ax.set_title(f'Mask {each}')
      #     ax.axis('off')

      #     ax = axes[1, each]
      #     plot_mask = Image.fromarray(np.transpose((images[each].numpy()*255).astype('uint8'),(1,2,0)))
      #     ax.imshow(plot_mask, interpolation='nearest', cmap='viridis', vmin = 0, vmax = 255)
      #     ax.set_title(f'Mask {each}')
      #     ax.axis('off')
      # # Anzeigen des Subplots
      # plt.tight_layout()
      # plt.show()

      predictions = model(images.to(device))

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
      # if leaf_noise_factor == 0:
      #   # Convert predicted logits to class labels
      #   predicted_batch_labels = predictions.argmax(dim=1).cpu().numpy()
      #   # Convert masks to class labels
      #   true_batch_labels = masks.squeeze(1).cpu().numpy()
      #   # Collect true and predicted labels
      #   true_labels.extend(true_batch_labels.flatten())
      #   predicted_labels.extend(predicted_batch_labels.flatten())

    losses_mean = losses/(batch_idx+1)

    # if leaf_noise_factor == 0:
    #   confusion = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_classes))
    #   recall = np.diag(confusion) / np.sum(confusion, axis=1)
    #   precision = np.diag(confusion) / np.sum(confusion, axis=0)
    #   accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
    #   iou_evaluator = MulticlassJaccardIndex(num_classes=num_classes, average=None)
    #   iou = iou_evaluator(torch.tensor(predicted_labels),torch.tensor( true_labels))

    #   # Prints to Terminal
    #   print('\n-------------------- Evaluation on Training Set --------------------')
    #   # print('Confusion Matrix:')
    #   # print(confusion)
    #   training_time = np.round(((time.time() - start_time)*100)/100)
    #   print(f"Epoch [{epoch+1}/{numEpochs}], Training Time Epoch: {int(training_time/(24*3600))}d {int(training_time/3600) % 24}h {int(training_time/60) % 60}min {int(training_time) % 60}s, Loss(mean):{losses_mean}, Loss(min):{min_train_loss}, Accuracy: {accuracy*100:.4f}, Recall: {recall*100}, Precision: {precision*100}, IoU: {iou*100}")

    #   # Tensorboard: Model Performance on Training Data:
    #   writer.add_scalars('IoU_Training',  {'Soil':iou[0], 'Plant':iou[1], 'Weed':iou[2]}, epoch)
    #   writer.add_scalars('Recall_Training',  {'Soil':recall[0], 'Plant':recall[1], 'Weed':recall[2]}, epoch)
    #   writer.add_scalars('Precision_Training',  {'Soil':precision[0], 'Plant':precision[1], 'Weed':precision[2]}, epoch)
    #   writer.add_scalar('Accuracy_Training',accuracy ,epoch)
    #   writer.add_scalars('Confusion_Matrix_Training',  {'SS':confusion[0][0], 'SP':confusion[1][0], 'SW':confusion[2][0], 'PP':confusion[1][1], 'PS':confusion[0][1], 'PW':confusion[2][1], 'WW':confusion[2][2], 'WS':confusion[0][2], 'WP':confusion[1][2]}, epoch) # True Value --> Predicted Value
    # # fig, ax = plt.subplots()
    # # ConfusionMatrixDisplay(confusion).plot(ax=ax)
    # # writer.add_figure("Confusion_Matrix_Training", fig, global_step=epoch)

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
      iou_evaluator = MulticlassJaccardIndex(num_classes=num_classes, average=None)
      iou = iou_evaluator(torch.tensor(val_predicted_labels),torch.tensor(val_true_labels))

      # Prints to Terminal
      print('\n-------------------- Evaluation on Validation Set --------------------')
      # print('Confusion Matrix:')
      # print(confusion)
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
    writer.add_scalars('Confusion_Matrix_Validation',{'SS':confusion[0][0], 'SP':confusion[1][0], 'SW':confusion[2][0], 'PP':confusion[1][1], 'PS':confusion[0][1], 'PW':confusion[2][1], 'WW':confusion[2][2], 'WS':confusion[0][2], 'WP':confusion[1][2]}, epoch) # True Value --> Predicted Value
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
