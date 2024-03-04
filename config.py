from pathlib import Path

def get_config():
    return {
        "batch_size":1,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 261,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel",
        'project_name': 'proj1'
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}"
    return str(Path('.') / model_folder / model_filename)


