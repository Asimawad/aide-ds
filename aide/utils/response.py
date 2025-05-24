import json
import re

import black


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []
    if "</think>" in text:
        parts = re.split(r"</think>", text, maxsplit=1, flags=re.DOTALL)
        # parts[0] is everything before </think>, parts[1] is everything after
        text = parts[1].strip() if len(parts) > 1 else ""

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    return format_code("\n\n".join(valid_code_blocks))


def extract_plan(text):
    """Extract plan from the text."""
    parsed_plan = []
    if "</think>" in text:
        parts = re.split(r"</think>", text, maxsplit=1, flags=re.DOTALL)
        # parts[0] is everything before </think>, parts[1] is everything after
        text = parts[1].strip() if len(parts) > 1 else ""

    # Extract everything after 'PLAN:' including multi-line content
    matches = re.findall(r"## Plan:\s*(.*)", text, re.DOTALL)
    for plan in matches:
        parsed_plan.append(plan.strip())

    # validate the parsed plan and format it
    # Combine and return the plan as a single string
    return "\n\n".join(parsed_plan)


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


# New: extract summary before PLAN
def extract_summary_and_plan(text, task=False):
    """Extract summary from the response before the 'PLAN:' section."""
    # Remove any thinking tags
    if "</think>" in text:
        parts = re.split(r"</think>", text, maxsplit=1, flags=re.DOTALL)
        text = parts[1].strip() if len(parts) > 1 else text
    # Split on PLAN: and return the first part as summary
    if task:
        return text
    if "PLAN" in text:
        parts = text.split("PLAN")
        summary = parts[0].strip()
        plan = parts[1].strip()
        return summary, plan
    return " " , text
