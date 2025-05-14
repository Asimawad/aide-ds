import wandb
import os
from .config import load_cfg
import pandas as pd
# --- Configuration ---
cfg = load_cfg()
from . import copytree
import os
import shutil
import wandb
from pathlib import Path

def copy_best_solution_and_submission():
    # Define the source and target directories
    workspaces_dir = os.path.join("workspaces", cfg.exp_name)
    logs_dir = Path(os.path.join("logs", cfg.exp_name) )

    # Define the specific folders to copy
    best_solution_dir = Path(os.path.join(workspaces_dir, "best_solution"))
    best_submission_dir = Path(os.path.join(workspaces_dir, "best_submission"))

    # Check if the directories exist and copy them to the logs directory
    if os.path.exists(best_solution_dir):
        copytree(best_solution_dir, logs_dir ,use_symlinks=False)
        print(f"Copied best_solution directory to {logs_dir}")
    else:
        print(f"best_solution directory not found in {workspaces_dir}")

    if os.path.exists(best_submission_dir):
        copytree(best_submission_dir, logs_dir,use_symlinks=False)
        print(f"Copied best_submission directory to {logs_dir}")
    else:
        print(f"best_submission directory not found in {workspaces_dir}")

def save_logs_to_wandb():
    # Ensure the logs directory is populated with the relevant folders (best_solution and best_submission)
    copy_best_solution_and_submission()

    # Save the entire logs directory to WandB
    print("Saving logs directory to WandB")
    wandb.save(f"logs/{cfg.exp_name}/*", base_path="./")  # Save log files

WANDB_ENTITY = "asim_awad"

WANDB_PROJECT = "MLE_BENCH" 

WANDB_RUN_NAME = cfg.wandb.run_name

DOWNLOAD_DIR = "./logs"

FILE_FILTER_PATTERN = cfg.wandb.run_name #solve me

def get_wb_data(project_name = "MLE_BENCH", wandb_run_name=WANDB_RUN_NAME , filter_pattern =FILE_FILTER_PATTERN, download_dir = "./" ):
        
    # --- Download Logic ---
    try:
        
        api = wandb.Api()

        # --- Find the run by name ---
        project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
        print(f"Searching for run named '{WANDB_RUN_NAME}' in project '{project_path}'...")

        
        filters = {"display_name": WANDB_RUN_NAME}
        

        runs = api.runs(project_path, filters=filters)

        if not runs:

            print(f"Error: No run found with the name '{WANDB_RUN_NAME}' in project '{project_path}'.")
            print("Please check the entity, project name, and run name.")
            exit() # Exit the script if the run isn't found
        


        if len(runs) > 1:
            print(f"Warning: Found {len(runs)} runs with the name '{WANDB_RUN_NAME}'. Using the first one found (ID: {runs[0].id}).")

        # Get the run object
        found_run = runs[0] 
        print(f"Successfully found run '{found_run.name}' ({found_run.id}). Proceeding to download files.")

        # Create the local download directory if it doesn't exist
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        print(f"Ensuring local download directory exists: {DOWNLOAD_DIR}")

        print(f"Listing files saved to the run's 'Files' tab...")
        # Get the list of files saved to this run's "Files" tab using the found_run object
        saved_files = found_run.files()
        try:
            hist = found_run.history(pandas=True)
            hist.to_csv(f"{DOWNLOAD_DIR}{wandb_file.name}/history.csv",index=False)
        except Exception as e:
            print(f"the history is not retrieved")
        download_count = 0

        for wandb_file in saved_files:
            # wandb_file.name is the path of the file as it appears in the W&B "Files" tab
            

            if wandb_file.name.startswith(FILE_FILTER_PATTERN):
                print(f"Found file to download: {wandb_file.name}")
                
                local_file_path = os.path.join(DOWNLOAD_DIR, wandb_file.name)
                
                # Ensure the local directory structure for this file exists
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                print(f"Downloading {wandb_file.name} to {local_file_path}...")

                # preserving the W&B file path structure within the root.
                wandb_file.download(root=DOWNLOAD_DIR,exist_ok =True)

                download_count += 1
                print("Download complete.")

        print(f"\nFinished downloading. Total files downloaded: {download_count}")
        print(f"Check the directory: {DOWNLOAD_DIR}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure:")
        print(f"- You have replaced WANDB_ENTITY, WANDB_PROJECT, and WANDB_RUN_NAME with correct values.")
        print("- You have run 'wandb login' or set the WANDB_API_KEY environment variable.")
        print("- The project exists and is accessible to your user.")
        print("- There is at least one run with the specified name in the project.")
        print("- The file filter pattern matches files that were actually saved.")