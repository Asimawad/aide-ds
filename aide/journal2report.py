from backend import query
from journal import Journal


def journal2report(journal: Journal, task_desc: dict):
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
        model="gemma2:2b", # edited by asim
        # model="gpt-4-turbo-preview",
        max_tokens=4096,
    )
