# run.py
import atexit
import logging
import shutil
import sys
import os
from pathlib import Path
from .utils import copytree,empirical_eval, advanced_metrics
from rich.logging import RichHandler      # add this import
os.environ['WANDB_API_KEY'] ="8ca0d241dd66f5a643d64a770d61ad066f937c48"
try:
    import wandb
    # wandb.init(project="aide-ds", entity="my-team")

except ImportError:
    wandb = None

from . import backend

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal, Node
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.markdown import Markdown
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from .utils.wandb_retreival import get_wb_data


import atexit, logging, os, shutil, sys
from pathlib import Path

from omegaconf import OmegaConf
from .utils.pretty_logging import logger           # <- sets up logging
from .utils.pretty_logging import log_step         # <- prints one-liner
from .utils import copytree, empirical_eval, advanced_metrics
from .utils.config import load_cfg, load_task_desc, prep_agent_workspace, save_run
from .utils.wandb_retreival import get_wb_data
from .journal import Journal
from .interpreter import Interpreter
from .agent import Agent
from .backend import compile_prompt_to_md

# WANDB -----------------------------------------------------------------
os.environ.setdefault("WANDB_API_KEY", "8ca0d241dd66f5a643d64a770d61ad066f937c48")
try:
    import wandb
except ImportError:     # still allow running without wandb installed
    wandb = None


# ────────────────────────────────────────────────────────────
def run() -> None:
    cfg = load_cfg()

    # mute very chatty libs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info(f'Starting run "{cfg.exp_name}"')
    cfg.log_dir.mkdir(parents=True, exist_ok=True)    # make sure it exists

    # ----------------------- WandB -------------------------------------
    wb_run = None
    if wandb and cfg.wandb.enabled:
        try:
            wb_run = wandb.init(
                project   = cfg.wandb.project,
                entity    = cfg.wandb.entity,
                name      = cfg.wandb.run_name,
                config    = OmegaConf.to_container(cfg, resolve=True),
                job_type  = "aide_run",
                tags      = ["aide-ds", cfg.agent.code.model],
            )
            logger.info(f"W&B run: {wb_run.url}")
        except Exception as e:
            logger.error(f"W&B init failed: {e}")

    # -------------------- task + workspace -----------------------------
    task_desc     = load_task_desc(cfg)
    task_desc_str = compile_prompt_to_md(task_desc)

    prep_agent_workspace(cfg)

    journal     = Journal()
    agent       = Agent(task_desc, cfg, journal, wandb_run=wb_run)
    interpreter = Interpreter(cfg.workspace_dir,
                              **OmegaConf.to_container(cfg.exec))  # type: ignore

    # ----------------------- main loop ---------------------------------
    global_step = 0
    try:
        while global_step < cfg.agent.steps:
            agent.step(exec_callback=interpreter.run,
                       current_step_number=global_step)

            save_run(cfg, journal)                # checkpoint
            latest = journal.nodes[-1]            # just finished node
            log_step(step      = global_step + 1,
                     total     = cfg.agent.steps,
                     stage     = getattr(latest, "stage", "?"),
                     is_buggy  = latest.is_buggy,
                     exec_time = getattr(latest, "exec_time", None),
                     metric    = (latest.metric.value
                                  if latest.metric and not latest.is_buggy else None)
                     )

            global_step = len(journal)

    finally:
        interpreter.cleanup_session()
        _finalise_wandb(wb_run, cfg, journal)
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir, ignore_errors=True)


# ----------------------------------------------------------------------
def _finalise_wandb(wb_run, cfg, journal):
    if not wb_run:
        return
    try:
        best = journal.get_best_node()
        if best:
            wandb.summary["best_validation_metric"] = best.metric.value
            wandb.summary["best_node_step"]         = best.step
        first_ok = next((n.step for n in journal.nodes if not n.is_buggy), None)
        wandb.summary["steps_to_first_working_code"] = first_ok if first_ok is not None else float("inf")

        if cfg.wandb.log_artifacts:
            bst_sub = cfg.workspace_dir / "best_submission"
            if bst_sub.exists():
                copytree(bst_sub, dst=cfg.log_dir, use_symlinks=True)
                wandb.save(f"{cfg.log_dir}/best_submission/*", base_path=str(cfg.log_dir))

        wb_run.finish()
    except Exception as e:
        logger.error(f"W&B finalisation failed: {e}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    run()





# class VerboseFilter(logging.Filter):
#     """
#     Filter (remove) logs that have verbose attribute set to True
#     """

#     def filter(self, record):
#         return not (hasattr(record, "verbose") and record.verbose)


# def journal_to_rich_tree(journal: Journal):
#     best_node = journal.get_best_node()

#     def append_rec(node: Node, tree):
#         if node.is_buggy:
#             s = "[red]◍ bug"
#         else:
#             style = "bold " if node is best_node else ""

#             if node is best_node:
#                 s = f"[{style}green]● {node.metric.value:.3f} (best)"
#             else:
#                 s = f"[{style}green]● {node.metric.value:.3f}"

#         subtree = tree.add(s)
#         for child in node.children:
#             append_rec(child, subtree)

#     tree = Tree("[bold blue]Solution tree")
#     for n in journal.draft_nodes:
#         append_rec(n, tree)
#     return tree


# def journal_to_string_tree(journal: Journal) -> str:
#     best_node = journal.get_best_node()
#     tree_str = "Solution tree\n"

#     def append_rec(node: Node, level: int):
#         nonlocal tree_str
#         indent = "  " * level
#         if node.is_buggy:
#             s = f"{indent}◍ bug (ID: {node.id})\n"
#         else:
#             # support for multiple markers; atm only "best" is supported
#             markers = []
#             if node is best_node:
#                 markers.append("best")
#             marker_str = " & ".join(markers)
#             if marker_str:
#                 s = f"{indent}● {node.metric.value:.3f} ({marker_str}) (ID: {node.id})\n"
#             else:
#                 s = f"{indent}● {node.metric.value:.3f} (ID: {node.id})\n"
#         tree_str += s
#         for child in node.children:
#             append_rec(child, level + 1)

#     for n in journal.draft_nodes:
#         append_rec(n, 0)

#     return tree_str

# def run():
#     cfg = load_cfg()

#     # ------------------------------------------------------------------ #
#     # 1.  Rich-style console logger + two file loggers
#     # ------------------------------------------------------------------ #
#     LOG_FORMAT_STD  = "[%(asctime)s] %(levelname)s: %(message)s"
#     LOG_FORMAT_RICH = "%(message)s"          # Rich prints its own time/level badge

#     logging.basicConfig(handlers=[], level=getattr(logging, cfg.log_level.upper()))

#     # mute very chatty libs
#     logging.getLogger("httpx").setLevel(logging.WARNING)

#     logger = logging.getLogger("aide")
#     cfg.log_dir.mkdir(parents=True, exist_ok=True)

#     # ---------- file logs -------------------------------------------------
#     file_handler = logging.FileHandler(cfg.log_dir / "aide.log", mode="w")
#     file_handler.setFormatter(logging.Formatter(LOG_FORMAT_STD))
#     file_handler.addFilter(VerboseFilter())              # hide verbose lines

#     verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log", mode="w")
#     verbose_file_handler.setFormatter(logging.Formatter(LOG_FORMAT_STD))

#     # ---------- pretty console log ---------------------------------------
#     console_handler = RichHandler(
#         rich_tracebacks=True,
#         markup=True,             # allow [colour] tags in messages
#         show_time=False,         # we already print time in file log
#         show_level=True,
#         show_path=False,
#     )
#     console_handler.setFormatter(logging.Formatter(LOG_FORMAT_RICH))
#     console_handler.addFilter(VerboseFilter())

#     # attach all three
#     logger.addHandler(file_handler)
#     logger.addHandler(verbose_file_handler)
#     logger.addHandler(console_handler)

#     logger.info(f'Starting run "{cfg.exp_name}"')

#     # ------------------------------------------------------------------ #
#     # 2.  (…everything else in run() stays exactly the same …)
#     # ------------------------------------------------------------------ #

#     wandb_run = None
#     if wandb and cfg.wandb.enabled:
#         try:
#             wandb_run = wandb.init(
#                 project=cfg.wandb.project,
#                 entity=cfg.wandb.entity, # Can be None
#                 name=cfg.wandb.run_name, # Can be None
#                 config=OmegaConf.to_container(cfg, resolve=True), # Log the config
#                 job_type="aide_run",
#                 tags=["aide-ds", cfg.agent.code.model] # Example tags
#             )
#             logger.info(f"W&B Run initialized: {wandb_run.url}")
#         except Exception as e:
#             logger.error(f"Failed to initialize W&B: {e}")
#             wandb_run = None # Ensure it's None if init fails

#     task_desc = load_task_desc(cfg)
#     task_desc_str = backend.compile_prompt_to_md(task_desc)

#     with Status("Preparing agent workspace (copying and extracting files) ..."):
#         prep_agent_workspace(cfg)

#     def cleanup():
#         if global_step == 0:
#             shutil.rmtree(cfg.workspace_dir)

#     atexit.register(cleanup)

#     journal = Journal()
#     agent = Agent(
#         task_desc=task_desc,
#         cfg=cfg,
#         journal=journal,
#         wandb_run=wandb_run

#     )

#     interpreter = Interpreter(
#         cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
#     )

#     global_step = len(journal)
#     prog = Progress(
#         TextColumn("[progress.description]{task.description}"),
#         BarColumn(bar_width=20),
#         MofNCompleteColumn(),
#         TimeRemainingColumn(),
#     )
#     status = Status("[green]Generating code...")
#     prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

#     def exec_callback(*args, **kwargs):
#         status.update("[magenta]Executing code...")
#         res = interpreter.run(*args, **kwargs)
#         status.update("[green]Generating code...")
#         return res

#     def generate_live():
#         tree = journal_to_rich_tree(journal)
#         prog.update(prog.task_ids[0], completed=global_step)

#         file_paths = [
#             f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
#             f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
#             f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
#         ]
#         left = Group(
#             Panel(Text(task_desc_str.strip()), title="Task description"),
#             prog,
#             status,
#         )

#         right = tree
#         wide = Group(*file_paths)

#         return Panel(
#             Group(
#                 Padding(wide, (1, 1, 1, 1)),
#                 Columns(
#                     [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
#                     equal=True,
#                 ),
#             ),
#             title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
#             subtitle="Press [b]Ctrl+C[/b] to stop the run",
#         )


#     # Main loop (Add try...finally to ensure wandb.finish)
#     try: # 
#         while global_step < cfg.agent.steps:


#             agent.step(exec_callback=exec_callback, current_step_number=global_step)


#             # on the last step, print the tree
#             if global_step == cfg.agent.steps - 1:
#                 logger.info(journal_to_string_tree(journal))
#             save_run(cfg, journal) # Save progress locally
#             global_step = len(journal)
#     finally: # Add finally block - This block runs no matter what.
#         # Clean up any other resources used by the program
#         interpreter.cleanup_session() 

#         # Check if a W&B run was successfully started
#         if wandb_run:
#             logger.info("Finishing W&B Run...")
#             try:
#                 # --- Log Summary Statistics ---
#                 # These appear in the 'Summary' section of your W&B run page.
#                 wo_step = None
#                 for node in journal.nodes:
#                     if not node.is_buggy:
#                         wo_step = node.step
#                         break # Found the first non-buggy node
                
#                 if wo_step is not None:
#                     wandb.summary["steps_to_first_working_code"] = wo_step
#                     logger.info(f"Logged Steps to First Working Code (WO): {wo_step}")
#                 else:
#                     wandb.summary["steps_to_first_working_code"] = float('inf') # Or a large number like cfg.agent.steps + 1
#                     logger.info("Logged Steps to First Working Code (WO): Never produced working code.")

#                 best_node = journal.get_best_node()
#                 if best_node:
#                     wandb.summary["best_validation_metric"] = best_node.metric.value
#                     wandb.summary["best_node_id"] = best_node.id
#                     wandb.summary["best_node_step"] = best_node.step

#                 # --- Log Artifacts (Consolidated Section) ---
#                 # This section collects various files/folders and logs them together.
#                 if cfg.wandb.log_artifacts: # Check if artifact logging is enabled
#                     # local_logs_dir
#                     bst_sub = cfg.workspace_dir / "best_submission"
#                     if bst_sub.exists():
#                         copytree(bst_sub,dst= cfg.log_dir, use_symlinks=True)
                    
#                     wandb.save(f"logs/{cfg.exp_name}/*", base_path="logs")
#                     wandb.save(f"workspaces/{cfg.exp_name}/best_submission/*",base_path="logs")

#                 # --- Finish the W&B Run ---

#                 wandb_run.finish()
#                 logger.info("W&B Run finished.")

#             except Exception as e_finalization:
#                 # Catch any errors during the overall W&B finalization process (summary, artifacts, finish)
#                 logger.error(f"Error during W&B finalization: {e_finalization}")
#                 if wandb_run: 
#                     # Try to finish the run one last time even if an error occurred
#                     # This prevents the run from staying in a "running" state indefinitely
#                     try:
#                          wandb_run.finish()
#                          logger.info("W&B Run finished after encountering an error.")
#                     except Exception as e_finish_retry:
#                          logger.error(f"Error during W&B finish retry: {e_finish_retry}")
#             try:
#                 get_wb_data()
#             except Exception as e:
#                 print(f"Couldnt get the wandb data:{e}")
#             try:
#                 empirical_eval.calculate_empirical_metrics()
#             except Exception as e:
#                 print(f"calculating empirical metrics gone wrong with this error : {e}")
#             try:
#                 advanced_metrics.calculate_advanced_metrics(cfg.exp_name, journal)
#             except Exception as e:
#                 print(f"calculating advanced metrics gone wrong with this error : {e}")

#         # Clean up workspace if this was the first step
#         if global_step == 0:
#             shutil.rmtree(cfg.workspace_dir)

# # Assuming 'run()' function exists elsewhere and contains the main program logic
# if __name__ == "__main__":
#     run()