"""configuration and setup utils"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Hashable, cast

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from aide.journal import Journal, filter_journal

from . import tree_export
from . import copytree, preproc_data, serialize,parse_model_id  
import re

shutup.mute_warnings()
logger = logging.getLogger("aide")


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


# <<< ADD WANDB CONFIG DATACLASS >>>
@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "MLE_BENCH"
    entity: str | None = None
    run_name: str | None = None
    log_code: bool = True
    log_artifacts: bool = True
@dataclass
class StageConfig:
    model: str
    temp: float
    max_new_tokens: int
    top_p: float = 1.0
    top_k: int = 40
    num_return_sequences: int = 1


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool
    convert_system_to_user: bool
    obfuscate: bool
    ITS_Strategy: str  # Can be "self-reflection" or "mcts"

    code: StageConfig
    feedback: StageConfig
    search: SearchConfig

    # # MCTS specific parameters
    # mcts_iterations: int = 10  # Number of MCTS iterations per step
    # mcts_exploration_weight: float = 1.414  # UCB exploration parameter
    # mcts_max_depth: int = 5  # Maximum tree depth
    # mcts_parallel_simulations: int = 1  # Number of parallel simulations


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str
    
    inference_engine:str
    
    exec: ExecConfig
    agent: AgentConfig
    wandb: WandbConfig


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    if "/" in cfg.agent.code.model:
        org, model = parse_model_id(cfg.agent.code.model)
        experiement_id = org+"_"+ model+"_"+ str(cfg.data_dir.name)+"_"+cfg.agent.ITS_Strategy +"_"+str(cfg.agent.steps)+"_steps"
    else:
        experiement_id = cfg.agent.code.model+"_"+ str(cfg.data_dir.name)+"_"+cfg.agent.ITS_Strategy +"_"+str(cfg.agent.steps)+"_steps"
    cfg.exp_name = cfg.exp_name or experiement_id # coolname.generate_slug(3)

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()
    
    # <<< ADD WANDB RUN NAME GENERATION (optional but good practice) >>>
    if cfg.wandb.enabled and cfg.wandb.run_name is None:
         cfg.wandb.run_name = cfg.exp_name # Use the coolname generated name
    
    # validate the config

    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal: Journal):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    filtered_journal = filter_journal(journal)
    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    # create the tree + code visualization
    # only if the journal has nodes
    if len(journal) > 0:
        tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
    # concatenate logs
    with open(cfg.log_dir / "full_log.txt", "w") as f:
        f.write(
            concat_logs(
                cfg.log_dir / "aide.log",
                cfg.workspace_dir / "best_solution" / "node_id.txt",
                cfg.log_dir / "filtered_journal.json",
            )
        )


def concat_logs(chrono_log: Path, best_node: Path, journal: Path):
    content = (
        "The following is a concatenation of the log files produced.\n"
        "If a file is missing, it will be indicated.\n\n"
    )

    content += "---First, a chronological, high level log of the AIDE run---\n"
    content += output_file_or_placeholder(chrono_log) + "\n\n"

    content += "---Next, the ID of the best node from the run---\n"
    content += output_file_or_placeholder(best_node) + "\n\n"

    content += "---Finally, the full journal of the run---\n"
    content += output_file_or_placeholder(journal) + "\n\n"

    return content


def output_file_or_placeholder(file: Path):
    if file.exists():
        if file.suffix != ".json":
            return file.read_text()
        else:
            return json.dumps(json.loads(file.read_text()), indent=4)
    else:
        return f"File not found."


# # Using regular expression to extract the competition name
# var = "/home/asim/Desktop/aide-ds/aide/example_tasks/house_prices"
# competition_name_regex = re.search(r'[^/]+$', var).group()
# print(competition_name_regex)  # Output: house_prices

# # Using pathlib to extract the competition name from a PosixPath
# var_path = Path(var)
# competition_name_pathlib = var_path.name
# print(competition_name_pathlib)  # Output: house_prices

# i  want to parse this var variable using regular expression in such a way that I only get the competetinon name. the last string after the last /
# also, assuming that this is not a string but rather a posix path, how can I achoev the same objective