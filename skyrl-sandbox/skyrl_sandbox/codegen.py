"""Code generation logic for SkyRL environments."""

import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from skyrl_sandbox.models import EnvironmentSpec


def get_template_dir() -> Path:
    """Get the templates directory."""
    return Path(__file__).parent / "templates"


def get_examples_dir() -> Path:
    """Get the examples directory in the SkyRL repo."""
    # Navigate up from skyrl-sandbox to the root, then to examples
    return Path(__file__).parent.parent.parent / "examples"


def setup_jinja_env() -> Environment:
    """Set up Jinja2 environment with templates."""
    template_dir = get_template_dir()
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def generate_env_code(spec: EnvironmentSpec) -> str:
    """Generate environment class code from spec."""
    jinja_env = setup_jinja_env()
    template = jinja_env.get_template("env_template.py.jinja2")
    return template.render(spec=spec)


def generate_main_code(spec: EnvironmentSpec) -> str:
    """Generate main entrypoint code from spec."""
    jinja_env = setup_jinja_env()
    template = jinja_env.get_template("main_template.py.jinja2")
    return template.render(spec=spec)


def export_environment(spec: EnvironmentSpec) -> tuple[bool, str, list[str]]:
    """
    Export environment code to the examples directory.
    
    Returns:
        (success, message, files_created)
    """
    try:
        # Get the target directory
        examples_dir = get_examples_dir()
        env_dir = examples_dir / spec.env_name
        
        # Create the directory
        env_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate code
        env_code = generate_env_code(spec)
        main_code = generate_main_code(spec)
        
        # Write files
        env_file = env_dir / "env.py"
        main_file = env_dir / f"main_{spec.env_name}.py"
        
        env_file.write_text(env_code)
        main_file.write_text(main_code)
        
        files_created = [
            str(env_file.relative_to(examples_dir.parent)),
            str(main_file.relative_to(examples_dir.parent))
        ]
        
        message = f"Successfully generated environment '{spec.env_name}' in {env_dir}"
        return True, message, files_created
        
    except Exception as e:
        return False, f"Error generating environment: {str(e)}", []


def preview_env_logic(
    spec: EnvironmentSpec,
    model_output: str,
    ground_truth: any,
    current_turn: int = 1
) -> dict:
    """
    Preview how the environment would handle a model output.
    
    This simulates the parsing and reward logic without actually
    creating an environment instance.
    """
    import re
    
    # Parse the answer
    parsed_answer = None
    if spec.parsing.method == "regex" and spec.parsing.pattern:
        match = re.search(spec.parsing.pattern, model_output)
        parsed_answer = match.group(1) if match else None
    elif spec.parsing.method == "json_path" and spec.parsing.json_path:
        try:
            import json
            from jsonpath_ng import parse as jsonpath_parse
            data = json.loads(model_output)
            jsonpath_expr = jsonpath_parse(spec.parsing.json_path)
            matches = jsonpath_expr.find(data)
            parsed_answer = matches[0].value if matches else None
        except Exception:
            parsed_answer = None
    
    found_answer = parsed_answer is not None
    
    # Calculate reward
    is_correct = False
    if spec.reward.method == "exact_match":
        is_correct = parsed_answer is not None and str(parsed_answer).strip() == str(ground_truth).strip()
    elif spec.reward.method == "regex_match":
        if parsed_answer is not None:
            is_correct = bool(re.match(str(ground_truth), str(parsed_answer)))
    elif spec.reward.method == "numeric_tolerance":
        try:
            if parsed_answer is not None:
                answer_num = float(parsed_answer)
                ground_truth_num = float(ground_truth)
                tolerance = spec.reward.tolerance or 0.01
                is_correct = abs(answer_num - ground_truth_num) <= tolerance
        except (ValueError, TypeError):
            is_correct = False
    
    # Determine reward
    if is_correct:
        reward = spec.reward.correct_reward
    elif found_answer and spec.reward.partial_reward is not None:
        reward = spec.reward.partial_reward
    else:
        reward = spec.reward.incorrect_reward
    
    # Determine if done
    if spec.env_type == "single_turn":
        done = True
    else:
        if spec.done_condition == "always_single_step":
            done = True
        elif spec.done_condition == "max_turns_only":
            done = current_turn >= spec.max_turns
        elif spec.done_condition == "correct_or_max_turns":
            done = current_turn >= spec.max_turns or is_correct
        else:
            done = False
    
    # Generate feedback
    feedback = None
    if not done:
        if found_answer and not is_correct:
            feedback = spec.feedback.on_incorrect.format(answer=parsed_answer)
        else:
            feedback = spec.feedback.on_format_error
    
    return {
        "parsed_answer": parsed_answer,
        "is_correct": is_correct,
        "reward": reward,
        "done": done,
        "feedback": feedback,
        "metadata": {
            "found_answer": found_answer,
            "current_turn": current_turn,
        }
    }

