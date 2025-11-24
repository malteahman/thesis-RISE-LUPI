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

def ensamble_model(config, writer_priv, writer_LUPI, writer_RGB):
    dataset_module = util.load_module(config.dataset.script_location)

    #val_set = dataset_module.val_set(config)
    #val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers,
    #                        pin_memory = True)
    
    test_set = dataset_module.test_set(config)
    test_loader = DataLoader(test_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)
    
    unet_base_priv = set_model(config, 'unet', n_channels=5)
    unet_base_priv_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-02-09_15-07-03', 'best_model.pth')
    unet_base_priv.load_state_dict(torch.load(unet_base_priv_model_path))

    unet_base_RGB = set_model(config, 'unet', n_channels=3)
    unet_base_RGB_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-02-09_15-09-09', 'best_model.pth')
    unet_base_RGB.load_state_dict(torch.load(unet_base_RGB_model_path))

    unet_base_only_priv = set_model(config, 'unet', n_channels=2)
    unet_base_only_priv_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-02-28_10-35-46', 'best_model.pth')
    unet_base_only_priv.load_state_dict(torch.load(unet_base_only_priv_model_path))

    zero_out_model = set_model(config, 'unet', n_channels=5) #förbättrar LUPI något
    zero_out_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-02-13_14-16-48', 'best_model.pth')
    zero_out_model.load_state_dict(torch.load(zero_out_model_path))
    zero_out_stats = [0.731, 0.994, 0.039, 0.956, 0.106]

    unet_pred_priv = set_model(config, 'unet_predict_priv', n_channels=3)
    unet_pred_priv_path = os.path.join('/raid/dlgroupmsc/logs/2024-03-13_11-00-37','best_model.pth')
    unet_pred_priv.load_state_dict(torch.load(unet_pred_priv_path)) 

    unet_priv_mtd = set_model(config, 'unet_mtd', n_channels=5)
    unet_priv_mtd_path = os.path.join('/raid/dlgroupmsc/logs/2024-02-21_14-28-39','best_model.pth')
    unet_priv_mtd.load_state_dict(torch.load(unet_priv_mtd_path)) 

    unet_ts_priv = set_model(config, 'unet', n_channels=5)
    unet_ts_priv_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-03-01_11-18-43', 'best_model.pth')
    unet_ts_priv.load_state_dict(torch.load(unet_ts_priv_model_path))
    unet_priv_stats = [0.778, 0.934, 0.114, 0.77, 0.189]

    unet_ts_RGB = set_model(config, 'unet', n_channels=3)
    unet_ts_RGB_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-03-04_11-25-09', 'best_model.pth')
    unet_ts_RGB.load_state_dict(torch.load(unet_ts_RGB_model_path))
    unet_RGB_stats = [0.752, 0.945, 0.106, 0.801, 0.182]

    unet_ts_LUPI = set_model(config, 'unet', n_channels=3)
    unet_ts_LUPI_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-03-07_13-47-35', 'best_model.pth')
    unet_ts_LUPI.load_state_dict(torch.load(unet_ts_LUPI_model_path))
    unet_LUPI_stats = [0.767, 0.879, 0.122, 0.711, 0.203]

    unet_ts_only_priv = set_model(config, 'unet', n_channels=3)
    unet_ts_only_priv_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-02-29_10-57-42', 'best_model.pth')
    unet_ts_only_priv.load_state_dict(torch.load(unet_ts_only_priv_model_path))
    unet_only_priv_stats = [0.751, 0.924, 0.124, 0.755, 0.187]

    unet_ts_LUPI_pred = set_model(config, 'unet_predict_priv', n_channels=3) #makes baseline slightly worse, but improves the other one
    unet_ts_LUPI_pred_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-04-05_10-50-50', 'best_model.pth')
    unet_ts_LUPI_pred.load_state_dict(torch.load(unet_ts_LUPI_pred_model_path))
    unet_LUPI_pred_stats = [0.763, 0.690, 0.152, 0.492, 0.157]

    unet_ts_LUPI_mtd = set_model(config, 'unet_mtd', n_channels=3)
    unet_ts_LUPI_mtd_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-04-10_10-12-37', 'best_model.pth')
    unet_ts_LUPI_mtd.load_state_dict(torch.load(unet_ts_LUPI_mtd_model_path))

    unet_ts_priv_mtd = set_model(config, 'unet_mtd', n_channels=5)
    unet_ts_priv_mtd_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-04-04_11-53-40', 'best_model.pth')
    unet_ts_priv_mtd.load_state_dict(torch.load(unet_ts_priv_mtd_model_path))

    transformer_ts_RGB = set_model(config, 'transformer', n_channels=3)
    transformer_ts_RGB_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-04-22_11-24-46', 'best_model.pth')
    transformer_ts_RGB.load_state_dict(torch.load(transformer_ts_RGB_model_path))

    transformer_ts_LUPI = set_model(config, 'transformer', n_channels=3)
    transformer_ts_LUPI_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-04-18_16-28-32', 'best_model.pth')
    transformer_ts_LUPI.load_state_dict(torch.load(transformer_ts_LUPI_model_path))

    transformer_ts_priv = set_model(config, 'transformer', n_channels=5)
    transformer_ts_priv_model_path = os.path.join('/raid/dlgroupmsc/logs/2024-04-23_09-52-13', 'best_model.pth')
    transformer_ts_priv.load_state_dict(torch.load(transformer_ts_priv_model_path))

    unet_base_priv.to(config.device)
    unet_base_priv.eval()

    unet_base_RGB.to(config.device)
    unet_base_RGB.eval()

    unet_base_only_priv.to(config.device)
    unet_base_only_priv.eval()

    zero_out_model.to(config.device)
    zero_out_model.eval()

    unet_pred_priv.to(config.device)
    unet_pred_priv.eval()

    unet_priv_mtd.to(config.device)
    unet_priv_mtd.eval()

    unet_ts_priv.to(config.device)
    unet_ts_priv.eval()

    unet_ts_RGB.to(config.device)
    unet_ts_RGB.eval()

    unet_ts_LUPI.to(config.device)
    unet_ts_LUPI.eval()

    unet_ts_only_priv.to(config.device)
    unet_ts_only_priv.eval()

    unet_ts_LUPI_pred.to(config.device)
    unet_ts_LUPI_pred.eval()

    unet_ts_LUPI_mtd.to(config.device)
    unet_ts_LUPI_mtd.eval()

    unet_ts_priv_mtd.to(config.device)
    unet_ts_priv_mtd.eval()

    transformer_ts_RGB.to(config.device)
    transformer_ts_RGB.eval()

    transformer_ts_LUPI.to(config.device)
    transformer_ts_LUPI.eval()

    transformer_ts_priv.to(config.device)
    transformer_ts_priv.eval()
    

    with torch.no_grad():
            zero_out_model = zero_out(1.0, zero_out_model)

    eval_loss_f = smp.losses.TverskyLoss(mode='multiclass')

    eval_loss_priv, eval_loss_LUPI, eval_loss_RGB = [], [], []
    #val_iter = iter(val_loader)
    test_iter = iter(test_loader)
    y_pred_list_priv, y_pred_list_LUPI, y_pred_list_RGB = [], [], []
    y_list = []
    num_models=16
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
            y_pred_base_priv = unet_base_priv(x)*0.5
            y_pred_base_RGB = unet_base_RGB(x[:,:3,:,:])*0.5
            y_pred_base_only_priv = unet_base_only_priv(x[:,3:,:,:])*0.5
            y_pred_zero_out = zero_out_model(x)*0.5
            y_pred_pred_priv,_ = unet_pred_priv(x[:,:3,:,:])
            y_pred_pred_priv = y_pred_pred_priv*0.5
            y_pred_priv_mtd = unet_priv_mtd(x, mtd)*0.5
            y_pred_ts_priv = unet_ts_priv(x)
            y_pred_ts_LUPI = unet_ts_LUPI(x[:,:3,:,:])
            y_pred_ts_RGB = unet_ts_RGB(x[:,:3,:,:])
            y_pred_ts_only_priv = unet_ts_only_priv(x[:,:3,:,:])
            y_pred_ts_LUPI_pred,_ = unet_ts_LUPI_pred(x[:,:3,:,:])
            y_pred_ts_LUPI_mtd = unet_ts_LUPI_mtd(x[:,:3,:,:], mtd)
            y_pred_ts_priv_mtd = unet_ts_priv_mtd(x, mtd)
            y_pred_transformer_RGB = transformer_ts_RGB(x[:,:3,:,:])
            y_pred_transformer_LUPI = transformer_ts_LUPI(x[:,:3,:,:])
            y_pred_transformer_priv = transformer_ts_priv(x)


        priv_predictions = [y_pred_base_priv, y_pred_base_RGB, y_pred_base_only_priv, y_pred_zero_out, 
                            y_pred_pred_priv, y_pred_priv_mtd, y_pred_ts_priv, y_pred_ts_LUPI, y_pred_ts_RGB,
                            y_pred_ts_only_priv, y_pred_ts_LUPI_pred, y_pred_ts_LUPI_mtd, y_pred_ts_priv_mtd, 
                            y_pred_transformer_RGB, y_pred_transformer_LUPI, y_pred_transformer_priv]
        LUPI_predictions = [y_pred_base_RGB, y_pred_zero_out, y_pred_pred_priv, y_pred_ts_LUPI, y_pred_ts_RGB,
                            y_pred_ts_only_priv, y_pred_ts_LUPI_pred, y_pred_ts_LUPI_mtd, y_pred_transformer_RGB, 
                            y_pred_transformer_LUPI] #10
        RGB_predictions = [y_pred_base_RGB, y_pred_ts_RGB, y_pred_transformer_RGB]
        
        
        #for i in range(len(predictions)):
        #    all_correct_probs[i], all_incorrect_probs[i], all_correct_sums[i], all_incorrect_sums[i], all_correct_stds[i], all_incorrect_stds[i] = confidence_count(
        #   
        #      predictions[i], y, all_correct_probs[i], all_incorrect_probs[i], all_correct_sums[i], all_incorrect_sums[i], all_correct_stds[i], all_incorrect_stds[i])


        softmaxed_preds_priv = [F.softmax(pred, dim=1) for pred in priv_predictions]
        softmaxed_preds_LUPI = [F.softmax(pred, dim=1) for pred in LUPI_predictions]
        softmaxed_preds_RGB = [F.softmax(pred, dim=1) for pred in RGB_predictions]

        y_pred_priv = torch.mean(torch.stack(softmaxed_preds_priv), dim=0)
        y_pred_LUPI = torch.mean(torch.stack(softmaxed_preds_LUPI), dim=0)
        y_pred_RGB = torch.mean(torch.stack(softmaxed_preds_RGB), dim=0)
        
        l_priv = eval_loss_f(y_pred_priv, y)
        l_LUPI = eval_loss_f(y_pred_LUPI, y)
        l_RGB = eval_loss_f(y_pred_RGB, y)
        eval_loss_priv.append(l_priv.item())
        eval_loss_LUPI.append(l_LUPI.item())
        eval_loss_RGB.append(l_RGB.item())

        y_pred_priv = torch.argmax(y_pred_priv, dim=1)
        y_pred_LUPI = torch.argmax(y_pred_LUPI, dim=1)
        y_pred_RGB = torch.argmax(y_pred_RGB, dim=1)

        y_pred_priv = y_pred_priv.to(torch.uint8).cpu().contiguous().numpy()
        y_pred_LUPI = y_pred_LUPI.to(torch.uint8).cpu().contiguous().numpy()
        y_pred_RGB = y_pred_RGB.to(torch.uint8).cpu().contiguous().numpy()
        y = y.to(torch.uint8).cpu().contiguous().numpy()
        y_pred_list_priv.append(y_pred_priv)
        y_pred_list_LUPI.append(y_pred_LUPI)
        y_pred_list_RGB.append(y_pred_RGB)
        y_list.append(y)

    l_test_priv = np.mean(eval_loss_priv)
    l_test_LUPI = np.mean(eval_loss_LUPI)
    l_test_RGB = np.mean(eval_loss_RGB)
    writer_priv.add_scalar('evaluation/loss', l_test_priv)
    writer_LUPI.add_scalar('evaluation/loss', l_test_LUPI)
    writer_RGB.add_scalar('evaluation/loss', l_test_RGB)
    miou_prec_rec_writing(config, y_pred_list_priv, y_list, 'evaluation', writer_priv, 0)
    miou_prec_rec_writing(config, y_pred_list_LUPI, y_list, 'evaluation', writer_LUPI, 0)
    miou_prec_rec_writing(config, y_pred_list_RGB, y_list, 'evaluation', writer_RGB, 0)

    #miou_prec_rec_writing_13(config, y_pred_list, y_list, 'evaluation', writer, 0)
    #conf_matrix(config, y_pred_list, y_list, writer, 0)
    
    # for correct_prob, incorrect_prob, correct, incorrect, correct_std, incorrect_std in zip(all_correct_probs, all_incorrect_probs, all_correct_sums, all_incorrect_sums, all_correct_stds, all_incorrect_stds):
    #     avg_correct_prob = sum(correct_prob) / len(correct_prob)
    #     avg_incorrect_prob = sum(incorrect_prob) / len(incorrect_prob)
    #     avg_correct_std = sum(correct_std) / len(correct_std)
    #     avg_incorrect_std = sum(incorrect_std) / len(incorrect_std)
    #     print('new model')
    #     print((correct / (correct + incorrect)).item())
    #     print(avg_correct_prob)
    #     print('avg correct std: ' + str(avg_correct_std))
    #     print(avg_incorrect_prob)
    #     print('avg incorrect std: ' + str(avg_incorrect_std))

   

@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
    log_dir_priv =  '/raid/dlgroupmsc/logs/ensamble_model_test_priv_1/tensorboard'
    log_dir_LUPI =  '/raid/dlgroupmsc/logs/ensamble_model_test_LUPI_1/tensorboard'
    log_dir_RGB = '/raid/dlgroupmsc/logs/ensamble_model_test_RGB_1/tensorboard'
    #log_dir = os.path.join(log_dir, 'tensorboard') #DON'T CHANGE, RISK OF OVERWRITING ALL PREVIOUSLY LOGGED DATA
    #print(log_dir)
    writer_priv = SummaryWriter(log_dir=log_dir_priv)
    writer_LUPI = SummaryWriter(log_dir=log_dir_LUPI)
    writer_RGB = SummaryWriter(log_dir=log_dir_RGB)
    ensamble_model(config, writer_priv, writer_LUPI, writer_RGB)
    #test.eval_on_test(config, writer, training_path)

if __name__ == '__main__':
    main()
