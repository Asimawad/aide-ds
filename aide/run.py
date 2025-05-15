import atexit
import logging
import shutil
import sys
import os
from pathlib import Path
from .utils import copytree, empirical_eval, advanced_metrics
from rich.logging import RichHandler
from tqdm import tqdm  # Import tqdm for progress bar

os.environ['WANDB_API_KEY'] = "8ca0d241dd66f5a643d64a770d61ad066f937c48"

try:
    import wandb
    from wandb.sdk.wandb_settings import Settings

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
from .utils.wandb_retreival import get_wb_data, save_logs_to_wandb

# Use the global logger
logger = logging.getLogger("aide")
logger.setLevel(logging.INFO)
logger.propagate = False

class VerboseFilter(logging.Filter):
    """
    Filter (remove) logs that have verbose attribute set to True
    """

    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def append_rec(node: Node, level: int):
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            # support for multiple markers; atm only "best" is supported
            markers = []
            if node is best_node:
                markers.append("best")
            marker_str = " & ".join(markers)
            if marker_str:
                s = f"{indent}● {node.metric.value:.3f} ({marker_str}) (ID: {node.id})\n"
            else:
                s = f"{indent}● {node.metric.value:.3f} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str

class WandbMirrorHandler(logging.Handler):
    def __init__(self, target_handler):
        # pick up the same level & formatter
        super().__init__(target_handler.level)
        self.target = target_handler
        self.setFormatter(target_handler.formatter)

    def emit(self, record):
        # 1) let your original handler do its thing
        self.target.emit(record)
        # 2) then send the *same* formatted message to wandb
        if wandb.run:  # make sure init() has happened
            msg = self.format(record)
            wandb.log({"log": msg}, step=wandb.run.step)



def run():
    logger.info("run.py has been triggered!")  # This will now work correctly
    
    cfg = load_cfg()
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )
    # don't want info logs from httpx
    httpx_logger: logging.Logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # save logs to files as well, using same format
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # we'll have a normal log file and verbose log file. Only normal to console
    file_handler = logging.FileHandler(cfg.log_dir / "aide.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log")
    verbose_file_handler.setLevel(logging.DEBUG)
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setLevel(level= cfg.log_level.upper())
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    # logger.addHandler(console_handler)

    logger.info(f'Starting run "{cfg.exp_name}"')

    wandb_run = None
    if wandb and cfg.wandb.enabled:
        try:
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity, 
                name=cfg.wandb.run_name, 
                dir="./",
                config=OmegaConf.to_container(cfg, resolve=True), # Log the config
                job_type="aide_run",
                tags=["aide-ds", cfg.agent.code.model], # Example tags

            )
            # wrap your existing handler
            wandb_mirror = WandbMirrorHandler(file_handler)
            wandb_mirror.setLevel(file_handler.level)
            logger.addHandler(wandb_mirror)
            # logger.info(f"W&B Run initialized: {wandb_run.url}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            wandb_run = None 
    
    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)
    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        wandb_run=wandb_run
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
    )

    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    try:
        # Wrap the loop with tqdm for a progress bar
        with tqdm(total=cfg.agent.steps, desc="Agent Steps", unit="step") as pbar:
            for i in range(cfg.agent.steps):
                agent.step(exec_callback=exec_callback, current_step_number=i+1)
                save_run(cfg, journal)  # Save progress locally
                global_step += 1
                pbar.update(1)
            # print(f"global_step: {global_step}")
            # print(f"cfg.agent.steps: {cfg.agent.steps}")
            # on the last step, print the tree
            if global_step == cfg.agent.steps - 1:
                logger.debug(journal_to_string_tree(journal))
                save_run(cfg, journal)

    finally:  # Add finally block - This block runs no matter what.
        interpreter.cleanup_session() 

        # # Check if a W&B run was successfully started
        if wandb_run:
            logger.info("Finishing W&B Run...")
            try:
                # --- Log Summary Statistics ---
                wo_step = None
                for node in journal.nodes:
                    if not node.is_buggy:
                        wo_step = node.step
                        break  # Found the first non-buggy node
                
                if wo_step is not None:
                    wandb.summary["steps_to_first_working_code"] = wo_step
                    logger.info(f"Logged Steps to First Working Code (WO): {wo_step}")
                else:
                    wandb.summary["steps_to_first_working_code"] = cfg.agent.steps + 10
                    logger.info("Logged Steps to First Working Code (WO): Never produced working code.")

                best_node = journal.get_best_node()
                if best_node:
                    wandb.summary["best_validation_metric"] = best_node.metric.value
                    wandb.summary["best_node_id"] = best_node.id
                    wandb.summary["best_node_step"] = best_node.step
                save_logs_to_wandb()
                wandb_run.finish()
            
                logger.info("W&B Run finished.")
            except Exception as e_finalization:
                logger.error(f"Error during W&B finalization: {e_finalization}")
                if wandb_run: 
                    try:
                         wandb_run.finish()
                         logger.info("W&B Run finished after encountering an error.")
                    except Exception as e_finish_retry:
                         logger.error(f"Error during W&B finish retry: {e_finish_retry}")


        # Clean up workspace if this was the first step
        if global_step == 0 or global_step == cfg.agent.steps:
            try:
                if global_step == cfg.agent.steps:
                    shutil.rmtree(cfg.workspace_dir/"input")
            except:
                shutil.rmtree(cfg.workspace_dir)

if __name__ == "__main__":
    run()
