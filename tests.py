
function_spec = {
    "name": "submit_review",
    "description": "Submit a review evaluating the output of the training script.",
    "parameters": {
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "True if execution failed or has a bug, otherwise false."
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "True if predictions are saved in a `submission.csv` file in the `./submission/` directory."
            },
            "summary": {
                "type": "string",
                "description": "Short summary (2-3 sentences) describing results or issues. No suggestions allowed."
            },
            "metric": {
                "type": "number",
                "description": "Validation metric if run was successful, otherwise null."
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "True if the metric should be minimized (e.g., MSE), false if maximized (e.g., accuracy)."
            }
        },
        "required": ["is_bug", "has_csv_submission", "summary", "metric", "lower_is_better"]
    }
}







messages= [{'role': 'system', 'content': '# Introduction\n\nYou are a Kaggle grandmaster attending a competition. You have written code to solve this task and now need to evaluate the output of the code execution. You should determine if there were any bugs as well as report the empirical findings.\n\n# Task description\n\n## Task goal\n\nPredict the sales price for each house\n\n## Task evaluation\n\nUse the RMSE metric between the logarithm of the predicted and observed values.\n\n# Implementation\n\n```python\nimport pandas as pd\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error\nimport numpy as np\n\n# Load data into a Pandas DataFrame\ndf = pd.read_csv("./input/train.csv")\ntest_data = pd.read_csv("./input/test.csv")  # load test data\n\n\n# Train the linear regression model\nmodel = LinearRegression()\nmodel.fit(df[["LotArea", "Neighborhood"]], df["SalePrice"])\n\n# Predict sale prices on the test set\npredictions = model.predict(test_data[["LotArea", "Neighborhood"]])\n\n# Evaluate the model using RMSE metric\nrmse = np.sqrt(mean_squared_error(test_data["SalePrice"], predictions))\nprint("RMSE:", rmse)\n\n\n# Save predictions in a CSV file for submission\nsubmission_df = pd.DataFrame({"Id": test_data["Id"], "SalePrice": predictions})\nsubmission_df.to_csv("./working/submission.csv", index=False)\n\n```\n\n# Execution output\n\n```\nTraceback (most recent call last):\n  File "runfile.py", line 13, in <module>\n    model.fit(df[["LotArea", "Neighborhood"]], df["SalePrice"])\n  File "/home/asim/.pyenv/versions/asim-vi/lib/python3.11/site-packages/sklearn/linear_model/_base.py", line 648, in fit\n    X, y = self._validate_data(\n           ^^^^^^^^^^^^^^^^^^^^\n  File "/home/asim/.pyenv/versions/asim-vi/lib/python3.11/site-packages/sklearn/base.py", line 584, in _validate_data\n    X, y = check_X_y(X, y, **check_params)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/asim/.pyenv/versions/asim-vi/lib/python3.11/site-packages/sklearn/utils/validation.py", line 1106, in check_X_y\n    X = check_array(\n        ^^^^^^^^^^^^\n  File "/home/asim/.pyenv/versions/asim-vi/lib/python3.11/site-packages/sklearn/utils/validation.py", line 879, in check_array\n    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)\n            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/asim/.pyenv/versions/asim-vi/lib/python3.11/site-packages/sklearn/utils/_array_api.py", line 185, in _asarray_with_order\n    array = numpy.asarray(array, order=order, dtype=dtype)\n            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/asim/.pyenv/versions/asim-vi/lib/python3.11/site-packages/pandas/core/generic.py", line 2084, in __array__\n    arr = np.asarray(values, dtype=dtype)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nValueError: could not convert string to float: \'CollgCr\'\nExecution time: a moment seconds (time limit is an hour).\n```\n'}] 


_tools=[{'type': 'function', 'function': {'name': 'submit_review', 'description': 'Submit a review evaluating the output of the training script.', 'parameters': {'type': 'object', 'properties': {'is_bug': {'type': 'boolean', 'description': 'true if the output log shows that the execution failed or has some bug, otherwise false.'}, 'has_csv_submission': {'type': 'boolean', 'description': 'true if the code saves the predictions on the test data in a `submission.csv` file in the `./submission/` directory, otherwise false. Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true. Otherwise, it should be evaluated as false. You can assume the ./submission/ directory exists and is writable.'}, 'summary': {'type': 'string', 'description': 'write a short summary (2-3 sentences) describing  the empirical findings. Alternatively mention if there is a bug or the submission.csv was not properly produced. DO NOT suggest fixes or improvements.'}, 'metric': {'type': 'number', 'description': 'If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.'}, 'lower_is_better': {'type': 'boolean', 'description': 'true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).'}}, 'required': ['is_bug', 'has_csv_submission', 'summary', 'metric', 'lower_is_better']}}, 'strict': True}]


_tool_choices = [{'type': 'function', 'function': {'name': 'submit_review'}}]


import openai
import time
_client = openai.OpenAI(base_url= 'http://localhost:11434/v1/' , api_key='ollama',max_retries=0)


def get_chat_response(messages, retries=5):
    for attempt in range(retries):
        try:
             
            response = _client.chat.completions.create(
                model= "llama3.2:latest", # Ensure you're using a model that supports function calling
                messages=messages,
                timeout=1000.00,
                tools=_tools,
                tool_choice= _tool_choices)
            print(response.choices[0])
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return None

output = get_chat_response(messages)
if output:
    print(output)
else:
    print("Failed to get a response after multiple attempts.")
