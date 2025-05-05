import wandb
run = wandb.init()
artifact = run.use_artifact('asim_awad/MLE_BENCH/')
artifact_dir = artifact.download()