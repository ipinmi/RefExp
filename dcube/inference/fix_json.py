import json
import re
import ast
from typing import List, Dict, Any


def parse_json(json_bbox: str) -> str:
    """
    Remove markdown fencing from the response.

    Args:
        json_bbox: Raw response potentially wrapped in markdown code blocks

    Returns:
        JSON string without markdown fencing
    """
    # Parsing out the markdown fencing
    lines = json_bbox.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_bbox = "\n".join(lines[i + 1 :])  # Remove everything before "```json"
            json_bbox = json_bbox.split("```")[
                0
            ]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found

    """# Also handle case where there's just ``` without json
    if "```" in json_bbox and "```json" not in json_bbox:
        # Remove simple code blocks
        json_bbox = json_bbox.replace("```", "")
    """

    return json_bbox.strip()


def fix_json_syntax(text: str) -> str:
    """
    Fix incomplete or malformed JSON syntax for bounding box data.

    Args:
        text: Possibly malformed JSON string

    Returns:
        Fixed JSON string
    """
    text = text.strip()

    # If it's already valid JSON, return it
    try:
        json.loads(text)
        return text
    except:
        pass

    fixed = text

    # CASE 1: Fix missing closing bracket in bbox array
    # Error: [{"bbox_2d": [156, 264, 482, 379, "label": "red sausage"}]
    # Expected: [{"bbox_2d": [156, 264, 482, 379], "label": "red sausage"}]
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*"label"'
    replacement = r'[\1, \2, \3, \4], "label"'
    fixed = re.sub(pattern, replacement, fixed)

    # CASE 2: Fix object structure
    # Error: "bbox_2d": [156, 264, 482, 379, "label": "red sausage"}]
    # Expected: [{"bbox_2d": [156, 264, 482, 379], "label": "red sausage"}]
    if '"bbox_2d"' in fixed and not fixed.strip().startswith("[{"):
        if not fixed.startswith("{"):
            fixed = "{" + fixed
        if not fixed.startswith("["):
            fixed = "[" + fixed

    # Count brackets and quotes to fix missing closures
    open_square = fixed.count("[")
    close_square = fixed.count("]")
    open_curly = fixed.count("{")
    close_curly = fixed.count("}")

    # CASE 4: Incomplete string quotes
    fixed = re.sub(r'("label":\s*"[^"]*?)(\}|\])', r'\1"\2', fixed)

    # CASE 5: Missing closing brackets
    if close_curly < open_curly:
        fixed += "}" * (open_curly - close_curly)
    if close_square < open_square:
        fixed += "]" * (open_square - close_square)

    # CASE 6: Double brackets issue
    fixed = re.sub(r"\{\{", "{", fixed)
    fixed = re.sub(r"\}\}", "}", fixed)

    # CASE 7: Verify the structure is a list of dictionaries
    if fixed.startswith("{") and not fixed.startswith("["):
        fixed = "[" + fixed + "]"

    # CASE 8: Incomplete JSON array
    # Error: "red sausage"}  (no closing square bracket)
    if fixed.endswith('"}') and open_square > close_square:
        fixed += "]"

    return fixed


def parse_bbox_response(response: str) -> List[Dict[str, Any]]:
    """
    Parse bounding box response with automatic markdown removal and syntax fixing.

    Args:
        response: Raw response string from model with/without markdown and syntax errors

    Returns:
        Parsed list of bounding box dictionaries (valid JSON)
    """
    # Remove markdown fencing
    cleaned_response = parse_json(response)

    # Try parsing to verify
    try:
        return json.loads(cleaned_response)
    except:
        pass

    try:
        return ast.literal_eval(cleaned_response)
    except:
        pass

    # If failed, fix syntax
    fixed_response = fix_json_syntax(cleaned_response)

    # Try parsing the fixed version
    try:
        return json.loads(fixed_response)
    except:
        pass

    try:
        return ast.literal_eval(fixed_response)
    except:
        pass

    # If all attempts fail, try a force fix
    try:
        # Extract the essential parts using regex
        bbox_pattern = r'\[?\s*\{[^}]*"bbox_2d"\s*:\s*\[([^\]]+)\][^}]*"label"\s*:\s*"([^"]+)"[^}]*\}[^}]*\]?'
        matches = re.findall(bbox_pattern, cleaned_response)

        if matches:
            result = []
            for coords, label in matches:
                coords_list = [int(x.strip()) for x in coords.split(",")]
                result.append({"bbox_2d": coords_list, "label": label})
            return result
    except:
        pass

    # try to extract any valid bbox data
    try:
        coord_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        label_pattern = r'"label"\s*:\s*"([^"]+)"'

        coords = re.findall(coord_pattern, cleaned_response)
        labels = re.findall(label_pattern, cleaned_response)

        if coords:
            result = []
            for i, coord in enumerate(coords):
                bbox_dict = {
                    "bbox_2d": [int(x) for x in coord],
                    "label": labels[i] if i < len(labels) else "parse_error",
                }
                result.append(bbox_dict)
            return result
    except:
        pass

    # If nothing works, return zero coordinates list with error indicator
    return [{"bbox_2d": [0, 0, 0, 0], "label": "parse_error"}]
