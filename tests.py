# def opt_messages_to_list(
#     system_message: str | None,
#     user_message: str | None,
#     convert_system_to_user: bool = False,
# ) -> list[dict[str, str]]:
#     messages = [{}]
#     if system_message:
#         if convert_system_to_user:
#             messages.append({"role": "user", "content": system_message})
#         else:
#             messages.append({"role": "system", "content": system_message})
#     if user_message:
#         messages.append(messages[1].update({"role": "user", "content": user_message}))
#     return messages
# x = opt_messages_to_list(system_message="hi sys" , user_message="hi user")
# print(x)
# import openai
# client = openai.OpenAI(api_key="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")

# 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

# # #  Basiaclly, this is waht we want as output: 
# # """output, req_time, in_tok_count, out_tok_count, info = query_func(
# #         system_message=system_message,
# #         user_message=user_message,
# #         func_spec=func_spec,
# #         convert_system_to_user=convert_system_to_user,
# #         **model_kwargs,
#     # )"""
# import logging
# import time

# from funcy import notnone, once, select_values

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
# @once
# def load_model(model_name):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     return tokenizer , model
# tokenizer , model = load_model(model_name=model_id)
# """Backend for HF API."""

# import json
# import logging
# import time

# from aide.backend.utils import (
#     FunctionSpec,
#     OutputType,
#     opt_messages_to_list,
#     backoff_create,
# )
# from funcy import notnone, once, select_values
# import openai

# logger = logging.getLogger("aide")

# _client: openai.OpenAI = None  # type: ignore


# OPENAI_TIMEOUT_EXCEPTIONS = (
#     openai.RateLimitError,
#     openai.APIConnectionError,
#     openai.APITimeoutError,
#     openai.InternalServerError,
# )

# @once
# def _setup_openai_client():
#     global _client
#     # Build the kwargs for the client; if any unsupported keys (e.g. "proxies") exist, remove them.
#     client_kwargs = {
#         "base_url": "https://router.huggingface.co/hf-inference/v1",
#         "api_key": "HF_TOKEN",
#     }
#     # Ensure no "proxies" key is passed.
#     client_kwargs.pop("proxies", None)
#     _client = openai.OpenAI(**client_kwargs)

# # @once
# # def _setup_openai_client():
# #     global _client
# #     _client = openai.OpenAI(
# #         base_url="https://router.huggingface.co/hf-inference/v1",
# #         api_key="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"
# #     )
# def query(
#     system_message: str | None,
#     user_message: str | None,
#     func_spec: FunctionSpec | None = None,
#     convert_system_to_user: bool = False,
#     **model_kwargs,
# ) -> tuple[OutputType, float, int, int, dict]:
#     _setup_openai_client()
#     filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

#     messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)

#     # if func_spec is not None:
#     #     filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
#     #     # force the model the use the function
#     #     filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

#     t0 = time.time()

#     completion = backoff_create(
#         _client.chat.completions.create,
#         OPENAI_TIMEOUT_EXCEPTIONS,
#         messages=messages,
#         **filtered_kwargs,
#     )
#     req_time = time.time() - t0

#     choice = completion.choices[0]
#     output = choice.message.content

#     # if func_spec is None:
#     #     output = choice.message.content
#     # else:
#     #     assert (
#     #         choice.message.tool_calls
#     #     ), f"function_call is empty, it is not a function call: {choice.message}"
#     #     assert (
#     #         choice.message.tool_calls[0].function.name == func_spec.name
#     #     ), "Function name mismatch"
#     #     try:
#     #         output = json.loads(choice.message.tool_calls[0].function.arguments)
#     #     except json.JSONDecodeError as e:
#     #         logger.error(
#     #             f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
#     #         )
#     #         raise e

#     in_tokens = completion.usage.prompt_tokens
#     out_tokens = completion.usage.completion_tokens

#     info = {
#         "system_fingerprint": completion.system_fingerprint,
#         "model": completion.model,
#         "created": completion.created,
#     }

#     return output, req_time, in_tokens, out_tokens, info
# """Backend for HF Inference API using the OpenAI client interface."""

# import json
# import logging
# import time

# from aide.backend.utils import (
#     FunctionSpec,
#     OutputType,
#     opt_messages_to_list,
#     backoff_create,
# )
# from funcy import notnone, once, select_values
# import openai

# logger = logging.getLogger("aide")

# _client: openai.OpenAI = None  # type: ignore

# OPENAI_TIMEOUT_EXCEPTIONS = (
#     openai.RateLimitError,
#     openai.APIConnectionError,
#     openai.APITimeoutError,
#     openai.InternalServerError,
# )

# @once
# def _setup_openai_client():
#     global _client
#     _client = openai.OpenAI(
#         base_url="https://router.huggingface.co/hf-inference/v1",
#         api_key="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"
#     )

# def query(
#     system_message: str | None,
#     user_message: str | None,
#     func_spec: FunctionSpec | None = None,
#     convert_system_to_user: bool = False,
#     **model_kwargs,
# ) -> tuple[OutputType, float, int, int, dict]:
#     _setup_openai_client()
#     # Remove None values from model_kwargs
#     filtered_kwargs: dict = select_values(notnone, model_kwargs)

#     # Prepare the messages list in the expected format.
#     messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)
    
#     # If function-calling is desired, add the extra parameters.
#     # if func_spec is not None:
#     #     filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
#     #     filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

#     # Set a default max_tokens if not provided.
#     if "max_tokens" not in filtered_kwargs:
#         filtered_kwargs["max_tokens"] = 500

#     t0 = time.time()
#     # Call the HF Inference API using the OpenAI client interface.
#     completion = backoff_create(
#         _client.chat.completions.create,
#         OPENAI_TIMEOUT_EXCEPTIONS,
#         model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         messages=messages,
#         **filtered_kwargs,
#     )
#     req_time = time.time() - t0

#     choice = completion.choices[0]

#     # if func_spec is None:
#     #     output = choice.message.content
#     # else:
#     #     # Handle function call responses.
#     #     assert choice.message.tool_calls, f"function_call is empty, it is not a function call: {choice.message}"
#     #     assert choice.message.tool_calls[0].function.name == func_spec.name, "Function name mismatch"
#     #     try:
#     #         output = json.loads(choice.message.tool_calls[0].function.arguments)
#     #     except json.JSONDecodeError as e:
#     #         logger.error(f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}")
#     #         raise e

#     in_tokens = completion.usage.prompt_tokens
#     out_tokens = completion.usage.completion_tokens

#     info = {
#         "system_fingerprint": completion.system_fingerprint,
#         "model": completion.model,
#         "created": completion.created,
#     }

#     return output, req_time, in_tokens, out_tokens, info

# # Example usage (for testing):
# if __name__ == "__main__":
#     # Prepare a simple prompt message.
#     test_messages = [{"role": "user", "content": "What is the capital of France?"}]
#     # Direct call without function specification.
#     result = backoff_create(
#         _client.chat.completions.create,
#         OPENAI_TIMEOUT_EXCEPTIONS,
#         model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         messages=test_messages,
#         max_tokens=500,
#     )
#     print(result.choices[0].message)



# import ollama






import json






import openai
# Step 2: Simulate the training script output
# 
# Define the prompt
prompt = "Review the execution log of my training script and call submit_review appropriately."
# messages = [
#     # {
#     #     "role": "user",
#     #     "content": "Please evaluate the output of the training script and provide a structured review."+training_output
#     # }

# {
#     "role": "system",
#     "content": "# Introduction\n\nYou are a Kaggle grandmaster attending a competition. In order to win this competition, you need to come up with an excellent and creative plan for a solution and then implement this solution in Python. We will now provide a description of the task.\n\n# Task description\n\n## Task goal\n\nPredict the sales price for each house\n\n## Task evaluation\n\nUse the RMSE metric between the logarithm of the predicted and observed values.\n\n# Memory\n\n\n\n# Instructions\n\n## Response format\n\nYour response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block.\n\n## Solution sketch guideline\n\n- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.\n- Take the Memory section into consideration when proposing the design, don't propose the same modelling solution but keep the evaluation the same.\n- The solution sketch should be 3-5 sentences.\n- Propose an evaluation metric that is reasonable for this task.\n- Don't suggest to do EDA.\n- The data is already prepared and available in the `./input` directory. There is no need to unzip any files.\n\n\n## Implementation guideline\n\n- <TOTAL_TIME_REMAINING: 23.0hrs 59.0mins 59.88644337654114secs>\n- <TOTAL_STEPS_REMAINING: 20>\n- The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,\n- **AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**\n- The code should be a single-file python program that is self-contained and can be executed as-is.\n- No parts of the code should be skipped, don't terminate the before finishing the script.\n- Your response should only contain a single code block.\n- Be aware of the running time of the code, it should complete within an hour.\n- All the provided input data is stored in \"./input\" directory.\n- **You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the \"./working\" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!\n- You can also use the \"./working\" directory to store any temporary files that your code needs to create.\n- REMEMBER THE ./submission/submission.csv FILE!!!!! The correct directory is important too.\n\n\n## Installed Packages\n\nYour solution can use any relevant machine learning packages such as: `torch`, `pandas`, `bayesian-optimization`, `torchvision`, `statsmodels`, `lightGBM`, `timm`, `xgboost`, `scikit-learn`, `torch-geometric`, `numpy`. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow.\n\n# Data Overview\n\n```\ninput/\n    data_description.txt (523 lines)\n    sample_submission.csv (1460 lines)\n    test.csv (1460 lines)\n    train.csv (1461 lines)\nsubmission/\n\nworking/\n```\n\n-> input/sample_submission.csv has 1459 rows and 2 columns.\nThe columns are: Id, SalePrice\n\n-> input/test.csv has 1459 rows and 80 columns.\nThe columns are: Id, MSSubClass, MSZoning, LotFrontage, LotArea, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2... and 65 more columns\n\n-> input/train.csv has 1460 rows and 81 columns.\nThe columns are: Id, MSSubClass, MSZoning, LotFrontage, LotArea, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2... and 66 more columns\n"
# }

# ]
# openai.api_key = "ollama"
# openai.base_url = 'http://localhost:11434/v1/'


# choice = response.choices[0]

# output = choice.message.content
# output = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
# print(response.choices[0].message.tool_calls)
# print(output)
# # Extract function call data
# function_call = response.get("message", {}).get("function_call")

# if function_call:
#     function_name = function_call["name"]
#     function_args = json.loads(function_call["arguments"])

#     print("\n✅ AI Function Call Response:")
#     print(f"Function Name: {function_name}")
#     print(f"Arguments: {json.dumps(function_args, indent=2)}")

#     # Process the function output as if it were a real function call
#     def process_review(response):
#         print("\n📌 Review Results:")
#         print("🔹 Bug Found:", response["is_bug"])
#         print("📂 Submission CSV Exists:", response["has_csv_submission"])
#         print("📝 Summary:", response["summary"])
#         print("📊 Metric Value:", response["metric"])
#         print("📉 Lower is Better:", response["lower_is_better"])

#     process_review(function_args)

# else:
#     print("\n❌ No function call was made by the model.")

# Define function specification for Ollama
training_output = """
# Epoch 1/10
# ----------
# Training loss: 0.523
# Validation loss: 0.478
# Validation RMSE: 0.691

# Epoch 2/10
# ----------
# Training loss: 0.432
# Validation loss: 0.412
# Validation RMSE: 0.642

# Training completed successfully.
# Predictions saved to ./submission/submission.csv
# """
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
