import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import csv
import os
import pandas as pd

import torch
from configs.hparams import hparams
from ehrdatasets.loader.datamodule import EhrDataModule
from ehrdatasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline, MlPipeline

def run_ml_experiment(config):
    los_config = get_los_info(f'/home/wangzixiang/retcare/ehr_datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(f'/home/wangzixiang/retcare/ehr_datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # logger
    checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    logger = CSVLogger(save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)
    L.seed_everything(config["seed"]) # seed for reproducibility

    # train/val/test
    pipeline = MlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=logger, num_sanity_val_steps=0)
    trainer.fit(pipeline, dm)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs

def run_dl_experiment(config):
    los_config = get_los_info(f'/home/wangzixiang/retcare/ehr_datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(f'/home/wangzixiang/retcare/ehr_datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # logger
    checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    if "time_aware" in config and config["time_aware"] == True:
        checkpoint_filename+="-ta" # time-aware loss applied
    logger = CSVLogger(save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)

    # EarlyStop and checkpoint callback
    if config["task"] in ["outcome", "readmission", "multitask"]:
        early_stopping_callback = EarlyStopping(monitor="auprc", patience=config["patience"], mode="max",)
        checkpoint_callback = ModelCheckpoint(filename="best", monitor="auprc", mode="max")
    elif config["task"] == "los":
        early_stopping_callback = EarlyStopping(monitor="mae", patience=config["patience"], mode="min",)
        checkpoint_callback = ModelCheckpoint(filename="best", monitor="mae", mode="min")

    L.seed_everything(config["seed"]) # seed for reproducibility

    # train/val/test
    trainer = L.Trainer(accelerator="gpu", devices=[1], max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback], num_sanity_val_steps=0)
    
    pipeline = DlPipeline(config)
    trainer.fit(pipeline, dm)
    best_model_path = checkpoint_callback.best_model_path
    
    pipeline = DlPipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs

if __name__ == "__main__":
    best_hparams = hparams # [TO-SPECIFY]
    for i in range(len(best_hparams)):
        config = best_hparams[i]
        run_func = run_ml_experiment if config["model"] in ["LR"] else run_dl_experiment
        seeds = [0] # [0,1,2,3,4]
        folds = [1]
        config["epochs"] = 50
        config["patience"] = 5
        for fold in folds:
            config["fold"] = fold
            for seed in seeds:
                config["seed"] = seed
                print(config)
                perf, outs = run_func(config)
                print(perf)
                pd.to_pickle(perf, f'logs/test/{config["dataset"]}/{config["model"]}/perf.pkl')