import torch
import torch.nn as nn
from loop2 import miou_prec_rec_writing, miou_prec_rec_writing_13, conf_matrix, save_image
from support_functions_noise import zero_out
from torch.utils.data import DataLoader
import util
import os
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp
import hydra
from support_functions_loop import set_model
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def confidence_count(y_pred, y, correct_probs, incorrect_probs, correct_sum, incorrect_sum, correct_std, incorrect_std):
    y_prob = F.softmax(y_pred,dim=1)
    predicted_class_probs = torch.max(y_prob, dim=1)[0]      

    y_pred = torch.argmax(y_pred, dim=1)

    correct_mask = (y_pred == y)
    incorrect_mask = ~correct_mask
    correct_sum += torch.sum(correct_mask)
    incorrect_sum += torch.sum(incorrect_mask) 

    if torch.sum(correct_mask) > 0:
        correct_avg_prob = torch.mean(predicted_class_probs[correct_mask])
        correct_std_prob = torch.std(predicted_class_probs[correct_mask])
        correct_avg_prob.to(torch.uint8).cpu().contiguous().numpy()
        correct_probs.append(correct_avg_prob.item()) 
        correct_std.append(correct_std_prob.item())

    if torch.sum(incorrect_mask) > 0:
        incorrect_avg_prob = torch.mean(predicted_class_probs[incorrect_mask])
        incorrect_std_prob = torch.std(predicted_class_probs[incorrect_mask])
        incorrect_avg_prob.to(torch.uint8).cpu().contiguous().numpy()
        incorrect_probs.append(incorrect_avg_prob.item())
        incorrect_std.append(incorrect_std_prob.item())

    return correct_probs, incorrect_probs, correct_sum, incorrect_sum, correct_std, incorrect_std

def ensamble_model(config, writer):
    dataset_module = util.load_module(config.dataset.script_location)

    val_set = dataset_module.val_set(config)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)
    
    test_set = dataset_module.test_set(config)
    test_loader = DataLoader(test_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)
    
    unet_priv = set_model(config, 'unet', n_channels=5)
    unet_priv_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-03-01_11-18-43', 'best_model.pth')
    unet_priv.load_state_dict(torch.load(unet_priv_model_path))
    unet_priv_stats = [0.778, 0.934, 0.114, 0.77, 0.189]

    unet_RGB = set_model(config, 'unet', n_channels=3)
    unet_RGB_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-03-04_11-25-09', 'best_model.pth')
    unet_RGB.load_state_dict(torch.load(unet_RGB_model_path))
    unet_RGB_stats = [0.752, 0.945, 0.106, 0.801, 0.182]

    unet_LUPI = set_model(config, 'unet', n_channels=3)
    unet_LUPI_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-03-07_13-47-35', 'best_model.pth')
    unet_LUPI.load_state_dict(torch.load(unet_LUPI_model_path))
    unet_LUPI_stats = [0.767, 0.879, 0.122, 0.711, 0.203]

    unet_LUPI_pred = set_model(config, 'unet_predict_priv', n_channels=3) #makes baseline slightly worse, but improves the other one
    unet_LUPI_pred_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-04-05_10-50-50', 'best_model.pth')
    unet_LUPI_pred.load_state_dict(torch.load(unet_LUPI_pred_model_path))
    unet_LUPI_pred_stats = [0.763, 0.690, 0.152, 0.492, 0.157]

    unet_mtd = set_model(config, 'unet_mtd', n_channels=5)
    unet_mtd_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-04-04_11-53-40', 'best_model.pth')
    unet_mtd.load_state_dict(torch.load(unet_mtd_model_path))
    unet_mtd_stats = [0.768, 0.921, 0.124, 0.743, 0.192]

    unet_pre_priv = set_model(config, 'unet', n_channels=5)
    unet_pre_priv_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-04-03_15-30-04', 'best_model.pth')
    unet_pre_priv.load_state_dict(torch.load(unet_pre_priv_model_path))
    unet_pre_priv_stats = [0.781, 0.932, 0.116, 0.760, 0.188]

    unet_only_priv = set_model(config, 'unet', n_channels=3)
    unet_only_priv_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-02-29_10-57-42', 'best_model.pth')
    unet_only_priv.load_state_dict(torch.load(unet_only_priv_model_path))
    unet_only_priv_stats = [0.751, 0.924, 0.124, 0.755, 0.187]

    base_priv = set_model(config, 'unet', n_channels=5)
    base_priv_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-02-09_15-07-03', 'best_model.pth')
    base_priv.load_state_dict(torch.load(base_priv_model_path))
    base_priv_stats = [0.747, 0.992, 0.044, 0.939, 0.124]

    zero_out_model = set_model(config, 'unet', n_channels=5) #förbättrar LUPI något
    zero_out_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-02-13_14-16-48', 'best_model.pth')
    zero_out_model.load_state_dict(torch.load(zero_out_model_path))
    zero_out_stats = [0.731, 0.994, 0.039, 0.956, 0.106]

    thresholds = torch.tensor([0.95, 0.95, 0.91, 0.69, 0.92, 0.95, 0.94, 1.1, 1.1]).to(config.device)
    thresholds = torch.tensor([0.998, 0.9997, 0.98, 0.90, 0.995, 0.9995, 0.9985, 1.1, 1.1]).to(config.device) 
    #not tried, adjusted according to prints
    

    unet_priv.to(config.device)
    unet_priv.eval()

    unet_RGB.to(config.device)
    unet_RGB.eval()

    unet_LUPI.to(config.device)
    unet_LUPI.eval()

    unet_LUPI_pred.to(config.device)
    unet_LUPI_pred.eval()

    unet_mtd.to(config.device)
    unet_mtd.eval()

    unet_pre_priv.to(config.device)
    unet_pre_priv.eval()

    unet_only_priv.to(config.device)
    unet_only_priv.eval()

    base_priv.to(config.device)
    base_priv.eval()

    zero_out_model.to(config.device)
    zero_out_model.eval()

    with torch.no_grad():
            zero_out_model = zero_out(1.0, zero_out_model)

    eval_loss_f = smp.losses.TverskyLoss(mode='multiclass')

    eval_loss = []
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)
    y_pred_list = []
    y_list = []
    num_models=5
    all_correct_probs = [[] for _ in range(num_models)]
    all_incorrect_probs = [[] for _ in range(num_models)]
    all_correct_stds = [[] for _ in range(num_models)]
    all_incorrect_stds = [[] for _ in range(num_models)]
    all_correct_sums = [0]*num_models
    all_incorrect_sums = [0]*num_models

    
    for batch in tqdm(test_iter):
        x, y, mtd, _ = batch

        x = x.to(config.device)
        y = y.to(config.device)
        mtd = mtd.to(config.device)

        with torch.no_grad():
            #y_pred_priv = unet_priv(x) #seccond best acc
            y_pred_RGB = unet_RGB(x[:,:3,:,:])
            y_pred_LUPI = unet_LUPI(x[:,:3,:,:])
            y_pred_LUPI_pred,_ = unet_LUPI_pred(x[:,:3,:,:]) #very low confidence levels, could be ramped up a bit
            #y_pred_mtd = unet_mtd(x,mtd)
            #y_pred_pre_priv = unet_pre_priv(x) #best acc
            y_pred_only_priv = unet_only_priv(x[:,:3,:,:])
            #y_pred_base_priv = base_priv(x)
            y_pred_zero_out = zero_out_model(x)

        #predictions = [y_pred_priv, y_pred_RGB, y_pred_LUPI, y_pred_LUPI_pred, y_pred_mtd, y_pred_pre_priv, y_pred_only_priv, y_pred_base_priv, y_pred_zero_out]
        predictions = [y_pred_RGB, y_pred_LUPI, y_pred_LUPI_pred, y_pred_only_priv, y_pred_zero_out]
        #softmaxed_preds = [F.softmax(pred, dim=1) for pred in predictions]
        
        for i in range(len(predictions)):
            all_correct_probs[i], all_incorrect_probs[i], all_correct_sums[i], all_incorrect_sums[i], all_correct_stds[i], all_incorrect_stds[i] = confidence_count(
                predictions[i], y, all_correct_probs[i], all_incorrect_probs[i], all_correct_sums[i], all_incorrect_sums[i], all_correct_stds[i], all_incorrect_stds[i])

        softmaxed_preds_LUPI = [F.softmax(pred, dim=1) for pred in [
        y_pred_RGB, y_pred_LUPI, y_pred_LUPI_pred, y_pred_only_priv, y_pred_zero_out*0.8
        ]]

        if False:
            y_pred_mean = torch.mean(torch.stack(softmaxed_preds), dim=0)
            thresholded_preds = [torch.where(pred >= threshold, pred, torch.tensor(float('nan')).to(config.device)) 
                     for pred, threshold in zip(softmaxed_preds, thresholds)] #9,2,19,512,512
            #print(thresholded_preds)
    
            isnan_tensor = torch.isnan(torch.stack(thresholded_preds))
            nan_counts = isnan_tensor.sum(dim=(1, 2, 3, 4))
            print((nan_counts-18*2*512*512)/(2*512*512))

            threshold_y_pred = torch.nanmean(torch.stack(thresholded_preds), dim=0)
            y_pred = torch.where(torch.isnan(threshold_y_pred), y_pred_mean, threshold_y_pred)
            

        else:
            y_pred = torch.mean(torch.stack(softmaxed_preds_LUPI), dim=0)
        
        l = eval_loss_f(y_pred, y)
        eval_loss.append(l.item())

        y_pred = torch.argmax(y_pred, dim=1)

        y_pred = y_pred.to(torch.uint8).cpu().contiguous().numpy()
        y = y.to(torch.uint8).cpu().contiguous().numpy()
        y_pred_list.append(y_pred)
        y_list.append(y)

    l_test = np.mean(eval_loss)
    print("loss: " + str(l_test))
    writer.add_scalar('evaluation/loss', l_test)
    miou_prec_rec_writing(config, y_pred_list, y_list, 'evaluation', writer, 0)
    miou_prec_rec_writing_13(config, y_pred_list, y_list, 'evaluation', writer, 0)
    #conf_matrix(config, y_pred_list, y_list, writer, 0)
    
    for correct_prob, incorrect_prob, correct, incorrect, correct_std, incorrect_std in zip(all_correct_probs, all_incorrect_probs, all_correct_sums, all_incorrect_sums, all_correct_stds, all_incorrect_stds):
        avg_correct_prob = sum(correct_prob) / len(correct_prob)
        avg_incorrect_prob = sum(incorrect_prob) / len(incorrect_prob)
        avg_correct_std = sum(correct_std) / len(correct_std)
        avg_incorrect_std = sum(incorrect_std) / len(incorrect_std)
        print('new model')
        print((correct / (correct + incorrect)).item())
        print(avg_correct_prob)
        print('avg correct std: ' + str(avg_correct_std))
        print(avg_incorrect_prob)
        print('avg incorrect std: ' + str(avg_incorrect_std))

   

@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
    log_dir =  '/raid/dlgroupmsc/logs/ensamble_model_test_LUPI'
    log_dir = os.path.join(log_dir, 'tensorboard') #DON'T CHANGE, RISK OF OVERWRITING ALL PREVIOUSLY LOGGED DATA
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    ensamble_model(config, writer)
    #test.eval_on_test(config, writer, training_path)

if __name__ == '__main__':
    main()
