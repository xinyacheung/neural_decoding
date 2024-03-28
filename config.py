from pathlib import Path

def get_config():
    return {
    "batch_size": 64,
    "stimuli_list":[i for i in range(16)],
    "neuron_ranges":[(0,140)],
    "num_neurons":140,
    "lr":1e-2,
    "num_epochs":50,
    "seed":1233,
    "pre_train":False,
    "device":"cuda:1",
    "model_folder":"weights",
    "model_basename":'train_16_',
    "loss_folder":'loss'
        }
    
def get_weights_file_path(config, epoch: str):
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / config["model_folder"] / model_filename)    
    
def latest_weights_file_path(config, from_epoch:str):
    if from_epoch == 'latest':
        model_filename = f"{config['model_basename']}*"
        weights_files = list(Path(config['model_folder']).glob(model_filename))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])
    else:
        model_filename = f"{config['model_basename']}{from_epoch}.pt"
        return  str(Path('.') / config["model_folder"] / model_filename)
    