import atexit
import logging
import shutil
import sys

try:
    import wandb
    wandb.init(project="asim_awad/aide-ds", entity="my-team")
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


def run():
    cfg = load_cfg()
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )
    # dont want info logs from httpx
    httpx_logger: logging.Logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("aide")
    # save logs to files as well, using same format
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # we'll have a normal log file and verbose log file. Only normal to console
    file_handler = logging.FileHandler(cfg.log_dir / "aide.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    logger.info(f'Starting run "{cfg.exp_name}"')
 # <<< INITIALIZE WANDB RUN >>>
    wandb_run = None
    if wandb and cfg.wandb.enabled:
        try:
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity, # Can be None
                name=cfg.wandb.run_name, # Can be None
                config=OmegaConf.to_container(cfg, resolve=True), # Log the config
                job_type="aide_run",
                tags=["aide-ds", cfg.agent.code.model] # Example tags
            )
            logger.info(f"W&B Run initialized: {wandb_run.url}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            wandb_run = None # Ensure it's None if init fails
    # <<< END INITIALIZE WANDB RUN >>>
    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

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

    global_step = len(journal)
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

    def generate_live():
        tree = journal_to_rich_tree(journal)
        prog.update(prog.task_ids[0], completed=global_step)

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]
        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"),
            prog,
            status,
        )
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    # while global_step <  cfg.agent.steps:    #PLEASE REVISE THIS HARCODED THING
    #     agent.step(exec_callback=exec_callback)
    #     # on the last step, print the tree
    #     if global_step == cfg.agent.steps - 1:
    #         logger.info(journal_to_string_tree(journal))
    #     save_run(cfg, journal)
    #     global_step = len(journal)
    # interpreter.cleanup_session()

    # Main loop (Add try...finally to ensure wandb.finish)
    try: # <<< Add try block
        while global_step < cfg.agent.steps:
            # <<< Use the live display >>>
            # Instead of live() context manager, update it inside the loop
            # if console_handler.level <= logging.INFO: # Check if INFO level is active for console
            #    prog.console.print(generate_live()) # Print the live view once per step

            # <<< Pass current step to agent.step for logging >>>
            agent.step(exec_callback=exec_callback, current_step_number=global_step)
            # <<< END Pass >>>

            # on the last step, print the tree
            if global_step == cfg.agent.steps - 1:
                logger.info(journal_to_string_tree(journal))
            save_run(cfg, journal) # Save progress locally
            global_step = len(journal)
    finally: # <<< Add finally block
        interpreter.cleanup_session()
        # <<< LOG FINAL RESULTS AND FINISH WANDB RUN >>>
        if wandb_run:
            logger.info("Finishing W&B Run...")
            try:
                best_node = journal.get_best_node()
                if best_node:
                    # Log best metric to summary
                    wandb.summary["best_validation_metric"] = best_node.metric.value
                    wandb.summary["best_node_id"] = best_node.id
                    wandb.summary["best_node_step"] = best_node.step

                    if cfg.wandb.log_artifacts:
                         # Log best solution code as artifact
                         best_code_path = cfg.log_dir / "best_solution.py"
                         if best_code_path.exists():
                              artifact_code = wandb.Artifact(f'solution_code_{wandb_run.id}', type='code')
                              artifact_code.add_file(str(best_code_path))
                              wandb_run.log_artifact(artifact_code)
                              logger.info(f"Logged best solution code artifact: {best_code_path}")

                         # Log best submission as artifact
                         best_submission_path = cfg.workspace_dir / "best_submission" / "submission.csv"
                         if best_submission_path.exists():
                              artifact_sub = wandb.Artifact(f'submission_{wandb_run.id}', type='submission')
                              artifact_sub.add_file(str(best_submission_path))
                              wandb_run.log_artifact(artifact_sub)
                              logger.info(f"Logged best submission artifact: {best_submission_path}")
                         else:
                              logger.warning("Best submission file not found for W&B artifact logging.")


                # Log the final journal as an artifact
                if cfg.wandb.log_artifacts:
                    journal_path = cfg.log_dir / "journal.json"
                    filtered_journal_path = cfg.log_dir / "filtered_journal.json"
                    if journal_path.exists():
                        artifact_journal = wandb.Artifact(f'journal_{wandb_run.id}', type='journal')
                        artifact_journal.add_file(str(journal_path))
                        if filtered_journal_path.exists():
                             artifact_journal.add_file(str(filtered_journal_path))
                        wandb_run.log_artifact(artifact_journal)
                        logger.info("Logged journal artifact.")

                wandb_run.finish()
                logger.info("W&B Run finished.")
            except Exception as e:
                logger.error(f"Error during W&B finalization: {e}")
                if wandb_run: # Try finishing again if error occurred before finish
                    wandb_run.finish()
        # <<< END LOG FINAL >>>

if __name__ == "__main__":
    run()
