import shutil
import logging
import random
import time
from typing import Any, Callable, cast
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code,trim_long_string
from .utils.self_reflection import perform_two_step_reflection  # Adjust path if needed
from pathlib import Path # <<< Add Path import

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger("aide")


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a `submission.csv` file in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        wandb_run=None
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        self.wandb_run = wandb_run

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    # @property
    # def _prompt_impl_guideline(self):
    #     tot_time_elapsed = time.time() - self.start_time
    #     tot_time_remaining = self.acfg.time_limit - tot_time_elapsed
    #     exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

    #     impl_guideline = [
    #         f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
    #         f"<TOTAL_STEPS_REMAINING: {self.acfg.steps - self.current_step}>",
    #         "The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,",
    #         "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE `./submission/` DIRECTORY.**",
    #         "The code should be a single-file python program that is self-contained and can be executed as-is.",
    #         "No parts of the code should be skipped, don't terminate the before finishing the script.",
    #         "Your response should only contain a single code block.",
    #         f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
    #         'All the provided input data is stored in "./input" directory.',
    #         '**You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the "./submission" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
    #         "You can use the `./working/` directory to store temporary files (e.g., models, intermediate data), but the final `submission.csv` MUST be in `./submission/`.",
    #         "REMEMBER THE `./submission/submission.csv` FILE!!!!! The correct directory is critical for evaluation.",
    #     ]
    #     if self.acfg.expose_prediction:
    #         impl_guideline.append(
    #             "The implementation should include a predict() function, "
    #             "allowing users to seamlessly reuse the code to make predictions on new data. "
    #             "The prediction function should be well-documented, especially the function signature."
    #         )

    #     if self.acfg.k_fold_validation > 1:
    #         impl_guideline.append(
    #             f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
    #         )

    #     return {"Implementation guideline": impl_guideline}
    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "1. Write a complete, single-file Python script. ",
            "2. starting with imports, and load necessary data from the './input/' directory.",
            "3. Implement the simple solution proposed in your plan.",
            "4. Calculate the evaluation metric on a validation set and **print it clearly** using a recognizable format, e.g., `print(f'Validation Metric: {metric_value}')`.",
            "5. **CRITICAL REQUIREMENT:** Generate predictions for the test data and save them EXACTLY to the path `./submission/submission.csv`. Ensure the file format matches the task description.",
            "6. The script must run without errors. Focus on correctness first.",
        ]
        return {"Implementation Guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""

        completion_text = None
        for _ in range(retries):
            if self.cfg.inference_engine == "HF" :
                completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                max_tokens=self.acfg.code.max_new_tokens,
                top_p=self.acfg.code.top_p,
                top_k=self.acfg.code.top_k,
                inference_engine = self.cfg.inference_engine,
                num_responses=self.acfg.code.num_return_sequences,
                convert_system_to_user=self.acfg.convert_system_to_user,
                )
            else:
                completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
            )
            # for debugging -> delete later

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "In order to win this competition, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview
        # print("____________________________________________________\n")
        # print(f"the currently used Prompt: {compile_prompt_to_md(prompt)}")
        # print("\n____________________________________________________\n")

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code)
        logger.info(f"Drafted new node {new_node.id}")
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should revise it in order to fix this. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "Your previous solution had a bug and/or did not produce a submission.csv, "
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def reflect(self, code: str) -> tuple[str, str]:
        """
        Performs a two-step self-reflection using the external utility function.

        Returns:
            Tuple: (reflection_plan, revised_code)
        """
        logger.info("Initiating two-step self-reflection...")
        reflection_plan, revised_code = perform_two_step_reflection(
            code=code,
            task_desc=self.task_desc,
            model_name=self.acfg.code.model,
            temperature=self.acfg.code.temp,
            convert_system_to_user=self.acfg.convert_system_to_user,
            query_func=query,  # Pass the imported query function
            wrap_code_func=wrap_code,  # Pass the imported wrap_code function
            extract_code_func=extract_code,  # Pass the imported extract_code function
        )

        if revised_code != code and revised_code:  # Check if code actually changed
            logger.info("Self-reflection resulted in code changes.")
        elif reflection_plan == "No specific errors found requiring changes.":
            logger.info("Self-reflection found no errors requiring changes.")
        else:
            logger.warning(
                "Self-reflection finished, but revised code is same as original or empty."
            )

        return reflection_plan, revised_code

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    # def step(self, exec_callback: ExecCallbackType):
    #     # clear the submission dir from previous steps
    #     shutil.rmtree(self.cfg.workspace_dir / "submission", ignore_errors=True)
    #     (self.cfg.workspace_dir / "submission").mkdir(exist_ok=True)

    #     if not self.journal.nodes or self.data_preview is None:
    #         self.update_data_preview()

    #     parent_node = self.search_policy()
    #     logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")
    #     draft_flag = False
    #     if parent_node is None:
    #         draft_flag = True
    #         result_node = self._draft()
    #     elif parent_node.is_buggy:
    #         result_node = self._debug(parent_node)
    #     else:
    #         result_node = self._improve(parent_node)

    #     # handle final cases where we missed buggy nodes somehow
    #     if not result_node.is_buggy:
    #         if not (self.cfg.workspace_dir / "submission" / "submission.csv").exists():
    #             result_node.is_buggy = True
    #             result_node.metric = WorstMetricValue()
    #             logger.info(
    #                 f"Actually, node {result_node.id} did not produce a submission.csv"
    #             )
    #     self.journal.append(result_node)

    #     # if the result_node is the best node, cache its submission.csv and solution.py
    #     # to best_solution/ by copying it there
    #     best_node = self.journal.get_best_node()
    #     if best_node is not None:
    #         if best_node.id == result_node.id:
    #             logger.info(f"Node {result_node.id} is the best node so far")
    #             best_solution_dir = self.cfg.workspace_dir / "best_solution"
    #             best_solution_dir.mkdir(exist_ok=True, parents=True)
    #             # copy submission/submission.csv to best_submission/submission.csv
    #             best_submission_dir = self.cfg.workspace_dir / "best_submission"
    #             best_submission_dir.mkdir(exist_ok=True, parents=True)
    #             shutil.copy(
    #                 self.cfg.workspace_dir / "submission" / "submission.csv",
    #                 best_submission_dir,
    #             )
    #             # copy solution.py and relevant node id to best_solution/
    #             with open(best_solution_dir / "solution.py", "w") as f:
    #                 f.write(result_node.code)
    #             # take note of the node id of the best node
    #             with open(best_solution_dir / "node_id.txt", "w") as f:
    #                 f.write(str(result_node.id))
    #         else:
    #             logger.info(f"Node {result_node.id} is not the best node")
    #             logger.info(f"Node {best_node.id} is still the best node")
    #     self.current_step += 1

#__________________________________________

    # <<< MODIFY step() method >>>
    def step(self, exec_callback: ExecCallbackType, current_step_number: int): # Add current_step_number
        # clear the submission dir from previous steps
        submission_dir = self.cfg.workspace_dir / "submission" # Define once
        shutil.rmtree(submission_dir, ignore_errors=True)
        submission_dir.mkdir(exist_ok=True)

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        parent_type = "None" if parent_node is None else parent_node.stage_name
        logger.info(f"Agent step {current_step_number}: Generating code (parent type: {parent_type})")

        draft_flag = False
        if parent_node is None:
            draft_flag = True
            result_node = self._draft()
            node_stage = "draft"
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
            node_stage = "debug"
        else:
            result_node = self._improve(parent_node)
            node_stage = "improve"

        # Log plan and initial code *before* reflection
        step_log_data = {
            f"{node_stage}/plan": result_node.plan,
            f"{node_stage}/parent_id": parent_node.id if parent_node else "None",
        }
        if self.wandb_run and self.cfg.wandb.log_code:
             # Limit code length for logging
             code_to_log = result_node.code[:10000] + ("\n..." if len(result_node.code) > 10000 else "")
             step_log_data[f"{node_stage}/initial_code"] = code_to_log



        # Apply reflection if applicable
        reflection_applied = False
        if draft_flag:  # Or based on your reflection strategy
            try:
                reflection_plan, reflection_code = self.reflect(code=result_node.code)
                if (
                    reflection_code
                    and reflection_code.strip()
                    and reflection_code != result_node.code
                ):
                    result_node.code = reflection_code
                    logger.info(
                        f"Node {result_node.id} self-reflected and updated code"
                    )
                    reflection_applied = True
                    # Log reflection results
                    step_log_data[f"{node_stage}/reflection_plan"] = reflection_plan
                    if self.wandb_run and self.cfg.wandb.log_code:
                         reflected_code_to_log = reflection_code[:10000] + ("\n..." if len(reflection_code) > 10000 else "")
                         step_log_data[f"{node_stage}/reflected_code"] = reflected_code_to_log

                elif reflection_plan != "No specific errors found requiring changes.":
                    logger.info(
                        f"Node {result_node.id} self-reflection completed, but no changes applied."
                    )
                    step_log_data[f"{node_stage}/reflection_plan"] = reflection_plan # Log even if no code change
                else:
                     step_log_data[f"{node_stage}/reflection_plan"] = "No errors found by reflection."


            except Exception as e:
                logger.error(
                    f"Error during self-reflection for node {result_node.id}: {e}",
                    exc_info=True,
                )
                step_log_data[f"{node_stage}/reflection_error"] = str(e)


        # Execute the potentially reflected code
        logger.info(f"Agent step {current_step_number}: Executing code for node {result_node.id} (stage: {node_stage}, reflection applied: {reflection_applied})")
        exec_start_time = time.time()
        exec_result = exec_callback(result_node.code, True) # reset_session=True usually
        exec_duration = time.time() - exec_start_time
        logger.info(f"Code execution finished in {exec_duration:.2f}s")

        # Parse execution result
        logger.info(f"Agent step {current_step_number}: Parsing execution results for node {result_node.id}")
        result_node = self.parse_exec_result(
            node=result_node, exec_result=exec_result
            ,)

        # Log execution and evaluation results
        step_log_data.update({
            f"exec/exec_time_s": result_node.exec_time,
            f"eval/is_buggy": result_node.is_buggy,
            f"exec/node_id": result_node.id,
            # Use trim_long_string for potentially long outputs
            f"exec/term_out": trim_long_string(result_node.term_out, threshold=2000, k=1000),
            f"eval/analysis": result_node.analysis,
        })
        if result_node.exc_type:
            step_log_data[f"exec/exception_type"] = result_node.exc_type
            step_log_data[f"exec/exception_info"] = str(result_node.exc_info) if result_node.exc_info else "None"

        if not result_node.is_buggy and result_node.metric and result_node.metric.value is not None:
            step_log_data[f"eval/validation_metric"] = result_node.metric.value
            step_log_data[f"eval/metric_maximize"] = result_node.metric.maximize
        else:
            # Log a placeholder if metric is invalid/buggy
            step_log_data[f"eval/validation_metric"] = float('nan') # W&B handles NaN well


        # Final check for submission file existence
        submission_path = submission_dir / "submission.csv"
        submission_exists = submission_path.exists()
        if not result_node.is_buggy and not submission_exists:
            result_node.is_buggy = True
            result_node.metric = WorstMetricValue()
            logger.info(
                f"Actually, node {result_node.id} did not produce a submission.csv"
            )
            # Update the logged data if W&B run exists
            if self.wandb_run:
                step_log_data[f"eval/is_buggy"] = True
                step_log_data[f"eval/validation_metric"] = float('nan')


        step_log_data[f"eval/submission_produced"] = submission_exists


        # --- Send log data to W&B ---
        if self.wandb_run:
            self.wandb_run.log(step_log_data, step=current_step_number)
        # --- End Send log data ---


        self.journal.append(result_node)

        # Log best solution artifacts *immediately* when a new best is found
        best_node = self.journal.get_best_node()
        if best_node is not None and best_node.id == result_node.id:
             logger.info(f"Node {result_node.id} is the best node so far (Metric: {best_node.metric.value:.4f})")
             best_solution_dir = self.cfg.workspace_dir / "best_solution"
             best_submission_dir = self.cfg.workspace_dir / "best_submission"
             best_solution_dir.mkdir(exist_ok=True, parents=True)
             best_submission_dir.mkdir(exist_ok=True, parents=True)

             if submission_exists:
                 shutil.copy(submission_path, best_submission_dir)
                 # <<< LOG BEST SUBMISSION ARTIFACT IMMEDIATELY >>>
                 if self.wandb_run and self.cfg.wandb.log_artifacts:
                     try:
                         artifact_sub = wandb.Artifact(f'best_submission', type='submission')
                         artifact_sub.add_file(str(best_submission_dir / "submission.csv"))
                         self.wandb_run.log_artifact(artifact_sub, aliases=["best", f"step_{current_step_number}"])
                         logger.info(f"Logged best submission artifact for step {current_step_number}")
                         # Also update summary metric in W&B
                         wandb.summary["best_validation_metric"] = best_node.metric.value
                         wandb.summary["best_node_id"] = best_node.id
                         wandb.summary["best_node_step"] = best_node.step
                     except Exception as e:
                          logger.error(f"Failed to log best submission artifact: {e}")
                 # <<< END LOG >>>
             else:
                  logger.warning(f"Best node {result_node.id} did not produce submission.csv, cannot cache/log artifact.")


             # Cache best solution code locally
             best_code_path = best_solution_dir / "solution.py"
             with open(best_code_path, "w") as f:
                 f.write(result_node.code)
             with open(best_solution_dir / "node_id.txt", "w") as f:
                 f.write(str(result_node.id))

             # <<< LOG BEST CODE ARTIFACT IMMEDIATELY >>>
             if self.wandb_run and self.cfg.wandb.log_artifacts:
                  try:
                       artifact_code = wandb.Artifact(f'best_solution_code', type='code')
                       artifact_code.add_file(str(best_code_path))
                       self.wandb_run.log_artifact(artifact_code, aliases=["best", f"step_{current_step_number}"])
                       logger.info(f"Logged best solution code artifact for step {current_step_number}")
                  except Exception as e:
                       logger.error(f"Failed to log best code artifact: {e}")
             # <<< END LOG >>>

        elif best_node:
             logger.info(f"Node {result_node.id} is not the best node (Best: {best_node.id} with metric {best_node.metric.value:.4f})")

        # self.current_step += 1 # Increment step counter at the end

#_____________________________________________________

 # <<< MODIFY parse_exec_result method >>>
    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        # --- Existing LLM call to evaluate execution ---
        # ... (introduction, prompt setup) ...
        introduction = (
             # ... (same as before) ...
        )
        if self.acfg.obfuscate:
             introduction = (
                 # ... (same as before) ...
             )
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        # Retry mechanism for the feedback LLM call (optional but good)
        max_retries = 3
        review_response = None
        for attempt in range(max_retries):
            try:
                review_response = cast(
                    dict,
                    query(
                        system_message=prompt,
                        user_message=None,
                        func_spec=review_func_spec,
                        model=self.acfg.feedback.model,
                        temperature=self.acfg.feedback.temp,
                        convert_system_to_user=self.acfg.convert_system_to_user,
                    ),
                )
                # Check if required keys are present
                if all(k in review_response for k in ["is_bug", "has_csv_submission", "summary", "metric", "lower_is_better"]):
                    break # Success
                else:
                    logger.warning(f"Feedback LLM response missing keys (attempt {attempt+1}/{max_retries}). Response: {review_response}")
                    review_response = None # Force retry
            except Exception as e:
                logger.error(f"Error querying feedback LLM (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error("Feedback LLM query failed after multiple retries.")
                    # Handle failure: maybe default to buggy?
                    review_response = {
                         "is_bug": True,
                         "has_csv_submission": False,
                         "summary": "Failed to get feedback from LLM.",
                         "metric": None,
                         "lower_is_better": True # Default assumption
                    }
                    break # Exit retry loop


        # if the metric isn't a float then fill the metric with the worst metric
        # <<< Ensure metric is None if not float, BEFORE checking existence >>>
        metric_value = review_response.get("metric") # Use .get for safety
        if not isinstance(metric_value, (float, int)):
            metric_value = None # Set to None if not a valid number
        # <<< END CHANGE >>>


        # do an extra check, to catch cases where judge fails
        submission_path = self.cfg.workspace_dir / "submission" / "submission.csv"
        has_csv_submission_actual = submission_path.exists()
        has_csv_submission_reported = review_response.get("has_csv_submission", False)


        node.analysis = review_response.get("summary", "Feedback LLM failed.") # Default value
        # Determine buggy status based on multiple factors
        node.is_buggy = (
            review_response.get("is_bug", True) # Default to True if key missing
            or node.exc_type is not None
            or metric_value is None # Use the validated metric_value
            or not has_csv_submission_reported # Judge's report
            or not has_csv_submission_actual # Actual file existence
        )

        if node.is_buggy:
            logger.info(
                f"Parsed results: Node {node.id} is buggy."
            )
            # Log reasons for being buggy
            bug_reasons = []
            if review_response.get("is_bug", True): bug_reasons.append("LLM judged buggy")
            if node.exc_type is not None: bug_reasons.append(f"Exception ({node.exc_type})")
            if metric_value is None: bug_reasons.append("Metric missing/invalid")
            if not has_csv_submission_reported: bug_reasons.append("LLM reported no submission")
            if not has_csv_submission_actual: bug_reasons.append("Submission file not found")
            logger.info(f"Buggy reasons: {'; '.join(bug_reasons)}")

            node.metric = WorstMetricValue()

            # <<< LOG BUGGY STATUS TO WANDB >>>
            if self.wandb_run:
                 bug_log = {"eval/buggy_reasons": "; ".join(bug_reasons)}
                 self.wandb_run.log(bug_log) # Log without step, will associate with last logged step
            # <<< END LOG >>>
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            node.metric = MetricValue(
                metric_value, maximize=not review_response.get("lower_is_better", True) # Default lower is better
            )

        return node













    # def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
    #     logger.info(f"Agent is parsing execution results for node {node.id}")

    #     node.absorb_exec_result(exec_result)

    #     introduction = (
    #         "You are a Kaggle grandmaster attending a competition. "
    #         "You have written code to solve this task and now need to evaluate the output of the code execution. "
    #         "You should determine if there were any bugs as well as report the empirical findings."
    #     )
    #     if self.acfg.obfuscate:
    #         introduction = (
    #             "You are an expert machine learning engineer attempting a task. "
    #             "You have written code to solve this task and now need to evaluate the output of the code execution. "
    #             "You should determine if there were any bugs as well as report the empirical findings."
    #         )
    #     prompt = {
    #         "Introduction": introduction,
    #         "Task description": self.task_desc,
    #         "Implementation": wrap_code(node.code),
    #         "Execution output": wrap_code(node.term_out, lang=""),
    #     }

    #     response = cast(
    #         dict,
    #         query(
    #             system_message=prompt,
    #             user_message=None,
    #             func_spec=review_func_spec,
    #             model=self.acfg.feedback.model,
    #             temperature=self.acfg.feedback.temp,
    #             convert_system_to_user=self.acfg.convert_system_to_user,
    #         ),
    #     )

    #     # if the metric isn't a float then fill the metric with the worst metric
    #     if not isinstance(response["metric"], float):
    #         response["metric"] = None

    #     # do an extra check, to catch cases where judge fails
    #     has_csv_submission = (
    #         self.cfg.workspace_dir / "submission" / "submission.csv"
    #     ).exists()

    #     node.analysis = response["summary"]
    #     node.is_buggy = (
    #         response["is_bug"]
    #         or node.exc_type is not None
    #         or response["metric"] is None
    #         or not response["has_csv_submission"]
    #         or not has_csv_submission
    #     )

    #     if node.is_buggy:
    #         logger.info(
    #             f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
    #         )
    #         node.metric = WorstMetricValue()
    #     else:
    #         logger.info(f"Parsed results: Node {node.id} is not buggy")
    #         node.metric = MetricValue(
    #             response["metric"], maximize=not response["lower_is_better"]
    #         )

    #     return node