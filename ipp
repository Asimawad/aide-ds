from setuptools import find_packages, setup
import atexit
import logging
import shutil
import sys

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


from backend import query
from journal import Journal
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
"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("aide")
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
from .utils.response import extract_code, extract_text_up_to_code, wrap_code
from .utils.self_reflection import perform_two_step_reflection  # Adjust path if needed

logger = logging.getLogger("aide")

from dataclasses import dataclass

from .backend import compile_prompt_to_md

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal, Node
from omegaconf import OmegaConf
from rich.status import Status
from .utils.config import load_task_desc, prep_agent_workspace, save_run, _load_cfg, prep_cfg
from pathlib import Path
import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger("aide")

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
from . import copytree, preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("aide")

"""
Contains functions to manually generate a textual preview of some common file types (.csv, .json,..) for the agent.
"""

import json
from pathlib import Path

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any

import numpy as np
from dataclasses_json import DataClassJsonMixin
import json
import re

import black

from typing import Callable
import re
import copy
import json
from pathlib import Path
from typing import Type, TypeVar

import dataclasses_json
from ..journal import Journal

"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph
from ..journal import Journal
import logging
from . import (
    backend_anthropic,
    backend_local,
    backend_openai,
    backend_vllm,
    backend_gdm,
    backend_deepseek,
)
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
"""Backend for Ollama model API."""

import json
import logging
import time
from funcy import notnone, once, select_values
import openai
from dotenv import load_dotenv
import os

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
# backend_local.py
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
"""Backend for OpenAI API."""

import json
import logging
import time
from dotenv import load_dotenv
import os
from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai
# aide/backend/backend_vllm.py
"""Backend for vLLM OpenAI-compatible API."""

import json
import logging
import time
import os
from funcy import notnone, once, select_values
import openai
from omegaconf import OmegaConf # To read config if needed

from aide.backend.utils import (FunctionSpec, OutputType, opt_messages_to_list, backoff_create)
import logging
from dataclasses import dataclass
from typing import Callable

import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

import backoff

this is for the interpreter environment:
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


        "HuggingFaceTB/SmolLM-1.7B-Instruct"

export MODEL_NAME="Qwen/Qwen2-0.5B-Instruct"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port 8000 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --gpu-memory-utilization 0.9