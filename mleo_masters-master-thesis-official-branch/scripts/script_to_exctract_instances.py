from torch.utils.data import DataLoader
from tqdm import tqdm
import util
import hydra
import torch
import numpy as np
import random
import json

def extract(config):
    relevant_classes = [4-1,9-1,12-1]
    dataset_module = util.load_module(config.dataset.script_location)
    train_set = dataset_module.train_set(config)
    
    train_loader = DataLoader(train_set, batch_size = 1, shuffle = True, num_workers = config.num_workers,
                              pin_memory = True)
  
    train_iter = iter(train_loader)
    instances_4 = []
    instances_9 = []
    instances_12 = []
 
    for batch in tqdm(train_iter):
        y_path, _, y, _, _ = batch
        
        #för varje bild, hitta masks som matchar de intressanta klasserna
        #spara alla masks, typ en lista per klasstyp
        #varje item i listan är en dict, med image path som key och sen en one hot encodad matris som value
        for c in relevant_classes:
            y_is_c = (y==c).int()
            if torch.all(y_is_c==0):
                continue
                
            y_is_c = torch.squeeze(y_is_c,0).tolist()
            if c == 4-1:
                instances_4.append({y_path[0]: y_is_c})
            elif c == 9-1:
                instances_9.append({y_path[0]: y_is_c})
            elif c == 12-1:
                instances_12.append({y_path[0]: y_is_c})

    print(len(instances_4))
    print(len(instances_9))
    print(len(instances_12))

    with open('/raid/dlgroupmsc/copy_and_paste/instances_4.txt', 'w') as file:
        json.dump(instances_4, file)

    with open('/raid/dlgroupmsc/copy_and_paste/instances_9.txt', 'w') as file:
        json.dump(instances_9, file)

    with open('/raid/dlgroupmsc/copy_and_paste/instances_12.txt', 'w') as file:
        json.dump(instances_12, file)

@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
    #log_dir =  '/raid/dlgroupmsc/logs/ensamble_model_threshold'
    
    extract(config)

if __name__ == '__main__':
    main()

#gå igneom listorna med global counter
    