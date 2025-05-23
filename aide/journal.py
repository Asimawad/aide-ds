"""
The journal is the core datastructure in AIDE that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin
from .interpreter import ExecutionResult
from .utils.metric import MetricValue
from .utils.response import trim_long_string


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)  # type: ignore
    summary: str = (None,)
    task_summary: str = " "
    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    code_quality: float = field(default=0, kw_only=True)  # type: ignore
    gold_medal: bool = field(default=0, kw_only=True)  # type: ignore
    silver_medal: bool = field(default=0, kw_only=True)  # type: ignore
    bronze_medal: bool = field(default=0, kw_only=True)  # type: ignore
    above_median: bool = field(default=0, kw_only=True)  # type: ignore
    effective_debug_step: bool = field(default=0, kw_only=True)  # type: ignore
    effective_reflections: bool = field(default=0, kw_only=True)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """
    A collection of nodes for an interaction session
    (when the agent interacts with a Jupyter notebook-like interface).
    """

    nodes: list[Node] = field(default_factory=list)
    completed: bool = False

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def generate_nb_trace(self, include_prompt, comment_headers=True) -> str:
        """Generate a trace of the interactive session in IPython format."""
        trace = []
        header_prefix = "## " if comment_headers else ""
        for n in self.nodes:
            trace.append(f"\n{header_prefix}In [{n.step+1}]:\n")
            trace.append(n.code)
            trace.append(f"\n{header_prefix}Out [{n.step+1}]:\n")
            trace.append(n.term_out)

        if include_prompt and self.nodes:
            trace.append(f"\n{header_prefix}In [{self.nodes[-1].step+2}]:\n")

        return "\n".join(trace).strip()


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)
    task_summary: str = None

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good=True) -> None | Node:
        """Return the best solution found so far (node with the highest validation metric)."""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        return max(nodes, key=lambda n: n.metric)


    def generate_summary(self, include_code: bool = True) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        if len(self.good_nodes) == 0:
            count = 0
            for n in self.draft_nodes:
                count += 1
                split_plan = n.plan.split("</think>")
                if len(split_plan) > 1:
                    summary_part = f"Design: {split_plan[1]}\n"
                else:
                    summary_part = f"Design: {n.plan}\n"
                summary.append(summary_part)
                if include_code:
                    summary.append(f"Code: {n.code}\n")
                if n.analysis:
                    summary.append(f"Results: {n.analysis}\n")
                if count == 3:
                    break
            return "\n-------------------------------\n".join(summary)
        else:

            for n in self.good_nodes:
                summary_part = f"Design: {n.plan}\n"
                split_plan = n.plan.split("</think>")
                if len(split_plan) > 1:
                    summary_part = f"Design: {split_plan[1]}\n"
                if include_code:
                    summary_part += f"Code: {n.code}\n"
                summary_part += f"Results: {n.analysis}\n"
                summary_part += f"Validation Metric: {n.metric.value}\n"
                summary.append(summary_part)
            return "\n-------------------------------\n".join(summary)


def get_path_to_node(journal: Journal, node_id: str) -> list[str]:
    path = [node_id]

    node2parent = {n.id: n.parent.id for n in journal.nodes if n.parent is not None}
    while node_id in node2parent:
        parent_id = node2parent[node_id]
        path.append(parent_id)
        node_id = parent_id
    return path[::-1]


def get_longest_path(journal: Journal) -> list[str]:
    longest_path = []
    for node in journal.nodes:
        path = get_path_to_node(journal, node.id)
        if len(path) > len(longest_path):
            longest_path = path
    return longest_path


def filter_on_path(journal: Journal, path: list[str]) -> Journal:
    journal_copy = copy.deepcopy(journal)
    journal_copy.nodes = [n for n in journal.nodes if n.id in path]
    # further filter nodes, setting their _term_out and exc_stack to "<OMITTED>"
    for n in journal_copy.nodes:
        n._term_out = "<OMITTED>"
        n.exc_stack = "<OMITTED>"

    return journal_copy


def filter_for_best_path(journal: Journal, best_node: str) -> Journal:
    path_to_best = get_path_to_node(journal, best_node)
    filtered_journal = filter_on_path(journal, path_to_best)
    return filtered_journal


def filter_for_longest_path(journal: Journal) -> Journal:
    longest_path = get_longest_path(journal)
    filtered_journal = filter_on_path(journal, longest_path)
    return filtered_journal


def filter_journal(journal: Journal) -> Journal:
    best_node = journal.get_best_node(only_good=True)

    if best_node is not None:
        filtered_journal = filter_for_best_path(journal, best_node.id)
    else:
        filtered_journal = filter_for_longest_path(journal)

    return filtered_journal


def journal2report(journal: Journal, task_desc: dict):
    from .backend import query

    """
    Generate a report from a journal, the report will be in markdown format.
    """
    report_input = journal.generate_summary(include_code=True)
    system_prompt_dict = {
        "Role": "You are a research assistant that always uses concise language.",
        "Goal": "The goal is to write a technical report summarising the empirical findings and technical decisions.",
        "Input": "You are given a raw research journal with list of design attempts and their outcomes, and a task description.",
        "Output": [
            "Your output should be a single markdown document.",
            "Your report should have the following sections: Introduction, Preprocessing, Modellind Methods, Results Discussion, Future Work",
            "You can include subsections if needed.",
        ],
    }
    context_prompt = (
        f"Here is the research journal of the agent: <journal>{report_input}<\\journal>, "
        f"and the task description is: <task>{task_desc}<\\task>."
    )
    return query(
        system_message=system_prompt_dict,
        user_message=context_prompt,
        model="o3-mini",
        max_tokens=4096,
    )
