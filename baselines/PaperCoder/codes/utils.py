import json
import re

try:
    from json_repair import repair_json as _repair_json
except ImportError:
    _repair_json = None


def _extract_json_object(text: str) -> str:
    """Extract the first complete JSON object from text, handling nested braces."""
    # Remove [CONTENT] wrappers
    text = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', text)
    # Find first {
    start = text.find('{')
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _escape_newlines_in_strings(s: str) -> str:
    """Replace literal newlines inside JSON string values with \\n."""
    result = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            escape_next = False
            result.append(ch)
            continue
        if ch == '\\' and in_string:
            escape_next = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == '\n':
            result.append('\\n')
            continue
        result.append(ch)
    return ''.join(result)


def _escape_latex_backslashes(s: str) -> str:
    """Escape LaTeX backslash commands (e.g. \\beta, \\bar) that JSON can't parse,
    while preserving valid JSON escape sequences."""
    valid_escapes = [
        (r'\\\\', '\x00'),   # \\ → placeholder
        (r'\\"', '\x01'),    # \" → placeholder
        (r'\\/', '\x02'),    # \/ → placeholder
        (r'\\b', '\x03'),    # \b → placeholder
        (r'\\f', '\x04'),    # \f → placeholder
        (r'\\n', '\x05'),    # \n → placeholder
        (r'\\r', '\x06'),    # \r → placeholder
        (r'\\t', '\x07'),    # \t → placeholder
    ]
    for pattern, placeholder in valid_escapes:
        s = s.replace(pattern, placeholder)
    # Now escape remaining backslashes (LaTeX commands like \beta, \alpha)
    s = s.replace('\\', '\\\\')
    # Restore valid JSON escapes
    for pattern, placeholder in valid_escapes:
        s = s.replace(placeholder, pattern)
    return s


def _clean_json_text(data: str) -> str:
    """Progressive cleanup to make JSON parseable."""
    # Remove [CONTENT] wrappers
    data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data)
    # Extract just the JSON object
    data = _extract_json_object(data)
    # Remove trailing commas before ] or }
    data = re.sub(r',\s*\]', ']', data)
    data = re.sub(r',\s*\}', '}', data)
    # Remove Python-style comments on value lines
    data = re.sub(r'(".*?"),\s*#.*', r'\1,', data)
    data = re.sub(r'(".*?")\s*#.*', r'\1', data)
    # Escape literal newlines inside JSON string values
    data = _escape_newlines_in_strings(data)
    # Strip control characters (except tabs)
    data = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', data)
    # Escape LaTeX backslashes
    data = _escape_latex_backslashes(data)
    return data


def extract_planning(trajectories_json_file_path):
    with open(trajectories_json_file_path) as f:
        traj = json.load(f)

    context_lst = []
    for turn in traj:
        if turn['role'] == 'assistant':
            content = turn['content']
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            context_lst.append(content)

    context_lst = context_lst[:3]
    return context_lst


def content_to_json(data):
    clean_data = _clean_json_text(data)
    try:
        return json.loads(clean_data)
    except json.JSONDecodeError:
        result = content_to_json2(data)
        if result is None and _repair_json is not None:
            result = _content_to_json_repair(data)
        return result


def content_to_json2(data):
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()
    clean_data = _extract_json_object(clean_data)
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)
    clean_data = re.sub(r',\s*\]', ']', clean_data)
    clean_data = re.sub(r',\s*\}', '}', clean_data)
    clean_data = _escape_newlines_in_strings(clean_data)
    clean_data = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', clean_data)
    clean_data = _escape_latex_backslashes(clean_data)
    try:
        return json.loads(clean_data)
    except json.JSONDecodeError:
        result = content_to_json3(data)
        if result is None and _repair_json is not None:
            result = _content_to_json_repair(data)
        return result


def _content_to_json_repair(data):
    """Last-resort: use json-repair library to fix malformed JSON."""
    if _repair_json is None:
        return None
    data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()
    try:
        repaired = _repair_json(data)
        return json.loads(repaired)
    except Exception:
        return None


def content_to_json3(data):
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()
    clean_data = _extract_json_object(clean_data)
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)
    clean_data = re.sub(r',\s*\]', ']', clean_data)
    clean_data = re.sub(r',\s*\}', '}', clean_data)
    clean_data = _escape_newlines_in_strings(clean_data)
    clean_data = re.sub(r'"""', '"', clean_data)
    clean_data = re.sub(r"'''", "'", clean_data)
    # Last resort: replace all backslashes
    clean_data = re.sub(r'\\', r'\\\\', clean_data)
    try:
        return json.loads(clean_data)
    except json.JSONDecodeError as e:
        print(e)
        return None 
    


def extract_code_from_content(content):
    pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
    code = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    if len(code) == 0:
        return ""
    else:
        return code[0]