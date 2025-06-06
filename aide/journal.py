"""
The journal is the core datastructure in AIDE that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""
# aide/journal.py

import copy
import time
import uuid
import logging # For potential warning in stage_name
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin
from .interpreter import ExecutionResult
from .utils.metric import MetricValue
from .utils.response import trim_long_string

from rich.console import Console

console = Console()

@dataclass(eq=False)
class Node(DataClassJsonMixin):
    # Positional fields:
    code: str  # Fields without defaults must come first.

    # 'id' is now a positional field with a default_factory.
    # This ensures its value from the original object is passed to __init__
    # during deepcopy's reconstruction via __reduce__.
    # For new Node() calls, if 'id' isn't provided, factory is used.
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # Keyword-only fields:
    plan: Optional[str] = field(default=None, kw_only=True)
    summary: Optional[str] = field(default=None, kw_only=True)
    task_summary: str = field(default=" ", kw_only=True)
    
    step: Optional[int] = field(default=None, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    
    # children is managed by __post_init__ and parent assignment.
    # init=False means it's not an __init__ param. repr=False is for cleaner default repr.
    children: set["Node"] = field(default_factory=set, init=False, repr=False)

    # stage is assigned by Agent logic (e.g. SelfDebugAgent)
    stage: Optional[str] = field(default=None, kw_only=True)

    # Execution info
    _term_out: Optional[list[str]] = field(default=None, kw_only=True)
    exec_time: Optional[float] = field(default=None, kw_only=True)
    exc_type: Optional[str] = field(default=None, kw_only=True)
    exc_info: Optional[dict] = field(default=None, kw_only=True)
    exc_stack: Optional[list[tuple]] = field(default=None, kw_only=True)

    # Evaluation
    analysis: Optional[str] = field(default=None, kw_only=True)
    metric: Optional[MetricValue] = field(default=None, kw_only=True)
    code_quality: float = field(default=0.0, kw_only=True)
    gold_medal: bool = field(default=False, kw_only=True)
    silver_medal: bool = field(default=False, kw_only=True)
    bronze_medal: bool = field(default=False, kw_only=True)
    above_median: bool = field(default=False, kw_only=True)
    effective_debug_step: bool = field(default=False, kw_only=True)
    effective_reflections: bool = field(default=False, kw_only=True)
    is_buggy: Optional[bool] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "draft" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        if self.parent.is_buggy is None:
            # This case should ideally not happen if parent nodes are fully evaluated.
            # Log a warning if it does. For calculation, treat as not buggy to avoid crashing.
            logging.warning(f"Parent node {self.parent.id} of node {self.id} has is_buggy=None. Defaulting stage logic.")
            return "improve" # Or handle as an error/default case
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
        if self._term_out is None:
            return "" 
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        # Ensure id exists before comparing.
        if not isinstance(other, Node) or not hasattr(other, 'id') or not hasattr(self, 'id'):
            return False
        if self.id is None or other.id is None: # Should not happen with proper init
             return False 
        return self.id == other.id

    def __hash__(self):
        if not hasattr(self, 'id') or self.id is None:
            # This should ideally be unreachable if __init__ (and deepcopy reconstruction) works.
            # If reached, it means 'id' was not set prior to hashing.
            # Raising a more specific error or logging can help debug Node construction.
            # For now, let original AttributeError propagate if id truly missing.
            # If self.id is None after factory, that's also an issue.
            # The default_factory for id should always return a str.
            raise AttributeError(f"CRITICAL: Node object (repr: {object.__repr__(self)}) is missing 'id' or 'id' is None at hashing time. This indicates a construction or deepcopy issue.")
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path (iterative calculation).
        - 0 if the node is not a debug node (parent is None or parent is not buggy)
        - n if there were n consecutive debugging steps up the parent chain.
        """
        if self.parent is None or self.parent.is_buggy is None or not self.parent.is_buggy:
            return 0
        
        # If we are here, self.parent exists and self.parent.is_buggy is True.
        # This node is the result of debugging its parent.
        depth = 0
        current_ancestor = self
        while current_ancestor.parent is not None and \
              current_ancestor.parent.is_buggy: # is_buggy should be True or False
            depth += 1
            current_ancestor = current_ancestor.parent
        return depth


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
# aide/journal.py -> Journal.generate_summary method

    def generate_summary(self, include_code: bool = True, max_nodes_to_summarize: int = 2, 
                         code_threshold: int = 1000, code_k: int = 400,
                         plan_threshold: int = 1000, plan_k: int = 400, # Added plan truncation
                         analysis_threshold: int = 500, analysis_k: int = 200) -> str:
        summary_parts = []
        nodes_to_consider = []

        # Prioritize good nodes, then recent draft nodes if no good nodes
        if self.good_nodes:
            sorted_good_nodes = sorted(self.good_nodes, key=lambda n: n.step, reverse=True)
            nodes_to_consider = sorted_good_nodes[:max_nodes_to_summarize]
        elif self.draft_nodes:
            sorted_draft_nodes = sorted(self.draft_nodes, key=lambda n: n.step, reverse=True)
            nodes_to_consider = sorted_draft_nodes[:max_nodes_to_summarize]
        
        if not nodes_to_consider and self.nodes: # Fallback to most recent nodes
            sorted_all_nodes = sorted(self.nodes, key=lambda n: n.step, reverse=True)
            nodes_to_consider = sorted_all_nodes[:max_nodes_to_summarize]

        for n_idx, n in enumerate(nodes_to_consider):
            current_entry_parts = [f"--- Attempt {n_idx+1} (Node ID: {n.id}, Stage: {n.stage_name}) ---"]
            plan_to_use = n.plan
            if n.plan and "</think>" in n.plan:
                split_plan = n.plan.split("</think>")
                if len(split_plan) > 1:
                    plan_to_use = split_plan[1]
            
            if plan_to_use:
                 current_entry_parts.append(f"Design: {trim_long_string(plan_to_use.strip(), threshold=plan_threshold, k=plan_k)}")
            else:
                 current_entry_parts.append("Design: No plan provided.")

            if include_code and n.code: # Check if n.code exists
                current_entry_parts.append(f"Code: {trim_long_string(n.code.strip(), threshold=code_threshold, k=code_k)}")
            elif include_code:
                current_entry_parts.append("Code: No code available.")

            if n.analysis:
                current_entry_parts.append(f"Results: {trim_long_string(n.analysis.strip(), threshold=analysis_threshold, k=analysis_k)}")
            if n.metric and n.metric.value is not None:
                current_entry_parts.append(f"Validation Metric: {n.metric.value:.4f} ({'Maximize' if n.metric.maximize else 'Minimize'})")
            if n.is_buggy:
                current_entry_parts.append(f"Outcome: Buggy (Error type: {n.exc_type or 'Unknown'})")
            
            summary_parts.append("\n".join(current_entry_parts))

        if not summary_parts:
            return "No previous attempts recorded or suitable for summary."
            
        return "\n\n-------------------------------\n".join(summary_parts)
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
