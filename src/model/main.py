import os
import config
from src.model.main_functions import Helper

if __name__ == "__main__":

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    config = {"mode": "train",
              "train_data_path": os.path.join(config.data_dir, "ME_train.txt"),
              "test_data_path":  os.path.join(config.data_dir, "ME_test.txt"),
              "analyze_data_path": os.path.join(config.data_dir, "sampling_data_5.txt"),
              "cache_dir_path": config.cache_dir,
              "model_dir_path": config.output_dir,
              "checkpoint": 0,
              "epoch": 50,
              "learning_rate": 0.0001,
              "dropout_rate": 0.1,
              "warmup_steps": 0,
              "max_grad_norm": 1.0,
              "batch_size": 12,
              "max_length": 200,
              "lstm_hidden": 1024,
              "lstm_num_layer": 1,
              "bidirectional_flag": True,
              "open_labels": 10,
              "close_labels": 10,
              "gradient_accumulation_steps": 1,
              "weight_decay": 0.0,
              "adam_epsilon": 1e-8
    }

    helper = Helper(config)

    if config["mode"] == "train":
        helper.train()
    elif config["mode"] == "test":
        helper.test()
    elif config["mode"] == "analyze":
        helper.analyze()
    elif config["mode"] == "demo":
        helper.demo()
