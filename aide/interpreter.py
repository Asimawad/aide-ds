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

logger = logging.getLogger("aide.interpreter")          # LOG+ narrower name
logger.setLevel(logging.DEBUG)                           # LOG+ raise to DEBUG for full trace


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [l for l in tb_lines if "aide/" not in l and "importlib" not in l]
        )
        tb_str = "".join([l for l in tb_lines])

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Process = None  # type: ignore

    def child_proc_setup(self, result_outq: Queue) -> None:
        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()
        os.chdir(str(self.working_dir))

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # remove the file after execution (otherwise it might be included in the data preview)
            os.remove(self.agent_file_name)

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)

        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()
        logger.debug(f"[spawn] New child PID={self.process.pid}")     # LOG+

    def cleanup_session(self):
        if self.process is None:
            return
        logger.debug(f"[cleanup] Terminating child PID={self.process.pid}")   # LOG+

        # give the child process a chance to terminate gracefully
        self.process.terminate()
        self.process.join(timeout=2)
        # kill the child process if it's still alive
        if self.process.exitcode is None:
            logger.warning("Child process failed to terminate gracefully, killing it..")
            self.process.kill()
            self.process.join()
        # don't wait for gc, clean up immediately
        self.process.close()
        self.process = None  # type: ignore
        logger.debug("[cleanup] Child cleaned up")                            
    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """

        logger.debug(f"REPL is executing code (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)
        logger.debug(f"[exec] Sent code to child (len={len(code.splitlines())} lines)")  # LOG+

        # wait for child to actually start execution (we don't want interrupt child setup)
        try:
            state = self.event_outq.get(timeout=10)
            assert state[0] == "state:ready", state
            logger.debug("[exec] Child signalled READY") 
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            queue_dump = ""
            while not self.result_outq.empty():
                queue_dump = self.result_outq.get()
                logger.error(f"REPL output queue dump: {queue_dump[:1000]}")
            self.cleanup_session()
            return ExecutionResult(
                term_out=[msg, queue_dump],
                exec_time=0,
                exc_type="RuntimeError",
                exc_info={},
                exc_stack=[],
            )
        assert state[0] == "state:ready", state
        start_time = time.time()

        # this flag indicates that the child ahs exceeded the time limit and an interrupt was sent

        child_in_overtime = False

        while True:
            try:
                # check if the child is done
                state = self.event_outq.get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                logger.debug(f"[exec] Child FINISHED (exc={state[1]})") # LOG+
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # we haven't heard back from the child -> check if it's still alive (assuming overtime interrupt wasn't sent yet)
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    queue_dump = ""
                    while not self.result_outq.empty():
                        queue_dump = self.result_outq.get()
                        logger.error(f"REPL output queue dump: {queue_dump[:1000]}")
                    self.cleanup_session()
                    return ExecutionResult(
                        term_out=[msg, queue_dump],
                        exec_time=0,
                        exc_type="RuntimeError",
                        exc_info={},
                        exc_stack=[],
                    )

                # child is alive and still executing -> check if we should sigint..
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:

                    logger.warning(
                        f"Child exceeded the {self.timeout}s timeout — "
                        "recycling interpreter session."
                    )

                    # ----- optional resource-pressure heuristics (keep) -----
                    try:
                        import psutil
                        proc        = psutil.Process(self.process.pid)
                        process_cpu = proc.cpu_percent(interval=0.5)
                        system_cpu  = psutil.cpu_percent(interval=0.5)
                        system_mem  = psutil.virtual_memory().percent

                        low_proc_cpu   = process_cpu < 5.0
                        sys_overloaded = (system_cpu > 90.0 or system_mem > 90.0)

                        if low_proc_cpu and sys_overloaded and running_time < self.timeout + 180:
                            logger.info(
                                "Likely resource contention; extending timeout by 3 min."
                            )
                            time.sleep(5)              # back-off, then re-check loop
                            continue                   # skip the recycle for now
                    except Exception as e:
                        logger.debug(f"psutil check failed: {e}")
                    # --------------------------------------------------------

                    # graceful SIGINT
                    try:
                        os.kill(self.process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass

                    self.process.join(timeout=2)

                    # hard kill if still alive
                    if self.process.is_alive():
                        logger.warning("Child ignored SIGINT, killing …")
                        self.process.kill()
                        self.process.join()

                    # clean up & recreate
                    self.cleanup_session()
                    self.create_process()
                    child_in_overtime = False        # reset flag
                    logger.info("[timeout] Session recycled; returning TimeoutError")      # LOG+

                    return ExecutionResult(
                        term_out=[
                            f"TimeoutError: Execution exceeded {self.timeout}s; "
                            "session recycled."
                        ],
                        exec_time=self.timeout,
                        exc_type="TimeoutError",
                        exc_info={},
                        exc_stack=[],
                    )

        output: list[str] = []
        # read all stdout/stderr from child up to the EOF marker
        # waiting until the queue is empty is not enough since
        # the feeder thread in child might still be adding to the queue
        
        # Check if this was a timeout situation where we killed the process
        if state[1] == "TimeoutError" and not self.process:
            # For timeouts where we killed the process, just collect what's available without waiting for EOF
            logger.warning("Timeout occurred and process was killed - collecting available output without waiting for EOF")
            while not self.result_outq.empty():
                try:
                    # Use a small timeout to avoid hanging if something goes wrong
                    output.append(self.result_outq.get(timeout=0.5))
                except queue.Empty:
                    break
        else:
            # Normal case - wait for the EOF marker
            while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
                try:
                    # Add a timeout here too for safety
                    output.append(self.result_outq.get(timeout=10))
                except queue.Empty:
                    logger.warning("Timed out waiting for EOF marker - proceeding anyway")
                    break
            
            # Only remove EOF marker if we got one
            if output and output[-1] == "<|EOF|>":
                output.pop()  # remove the EOF marker

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        # logger.info(f"[return] exec_time={exec_time:.2f}s  exc={e_cls_name}")   # LOG+
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
