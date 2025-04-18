#!/bin/bash
source /aide-ds/.aide-ds/bin/activate
export TOKENIZERS_PARALLELISM=false

ollama pull deepseek-r1:32B 2>/dev/null
echo "Current working directory: $(pwd)"
export LLM="deepseek-r1:32B"    
aide data_dir="example_tasks/house_prices" \
    goal="Predict the sales price for each house" \
    eval="Use the RMSE metric between the logarithm of the predicted and observed values."\
    exec.timeout=120 \
    agent.code.model=$LLM

# Capture the exit code of the aide command
EXIT_CODE=$? 

# Check if the aide command completed successfully (exit code 0)
if [ $EXIT_CODE -eq 0 ]; then
  echo "Aide command completed successfully. Copying outputs to Aichor path..."

    # --- DEBUGGING: Check if variables and paths exist ---
  echo "AICHOR_OUTPUT_PATH is set to: $AICHOR_OUTPUT_PATH"

  echo "Checking if source ./logs directory exists..."
  ls -ld ./logs

  echo "Checking if source ./workspaces directory exists..."
  ls -ld ./workspaces

  # Copy the contents of the logs abd workspaces directory 
  echo "Copying ./logs/* & ./workspaces/*  to $AICHOR_OUTPUT_PATH/"
  .aide-ds/bin/python upload_results.py
  
  # --- DEBUGGING: List contents of output path AFTER copy ---
  echo "Contents of AICHOR_OUTPUT_PATH after copy:"
  ls -lR "$AICHOR_OUTPUT_PATH"

  echo "Outputs copied to specified local path for Aichor upload."
else
  echo "Aide command failed with exit code $EXIT_CODE. Skipping output copy."
  exit $EXIT_CODE # Ensure the Aichor job also fails
fi
# --- END: New lines to add --- 

echo "run.sh finished."
exit 0