"""Code generation logic and preview utilities for SkyRL environments."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jsonpath_ng import parse as jsonpath_parse

from skyrl_sandbox.models import EnvironmentSpec, LLMVerifierConfig, RewardCreationConfig, RewardSchemeConfig


def get_template_dir() -> Path:
    """Get the templates directory."""
    return Path(__file__).parent / "templates"


def get_runs_dir() -> Path:
    """Get the generated runs directory under skyrl-train."""
    return Path(__file__).parent.parent.parent / "skyrl-train" / "runs"


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
    return template.render(spec=spec, spec_dict=spec.model_dump())


def generate_register_code(spec: EnvironmentSpec) -> str:
    """Generate register entrypoint code from spec."""
    jinja_env = setup_jinja_env()
    template = jinja_env.get_template("register_env_template.py.jinja2")
    return template.render(spec=spec, spec_dict=spec.model_dump())


def export_environment(spec: EnvironmentSpec) -> tuple[bool, str, list[str]]:
    """
    Export environment code to the runs directory for custom environments.
    
    Returns:
        (success, message, files_created)
    """
    try:
        repo_root = Path(__file__).parent.parent.parent
        runs_dir = get_runs_dir()

        env_dir = runs_dir / spec.env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        env_code = generate_env_code(spec)
        register_code = generate_register_code(spec)

        env_file = env_dir / "env.py"
        register_file = env_dir / "register_env.py"

        env_file.write_text(env_code)
        register_file.write_text(register_code)

        files_created = [
            str(env_file.relative_to(repo_root)),
            str(register_file.relative_to(repo_root)),
        ]

        message = f"Successfully generated environment '{spec.env_name}'"
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
    parsed_answer = _parse_answer(spec.parsing, model_output)
    reward_signal = _evaluate_reward_signal(
        spec.reward.creation, parsed_answer, model_output, ground_truth
    )
    reward_value = _apply_reward_scheme(spec.reward.scheme, reward_signal)
    done = _determine_done(spec, reward_signal["is_correct"], current_turn)

    feedback = None
    if spec.env_type == "multi_turn" and not done:
        if reward_signal["found_answer"] and not reward_signal["is_correct"]:
            feedback = spec.feedback.on_incorrect.format(answer=parsed_answer)
        else:
            feedback = spec.feedback.on_format_error

    return {
        "parsed_answer": parsed_answer,
        "is_correct": reward_signal["is_correct"],
        "reward": reward_value,
        "done": done,
        "feedback": feedback,
        "metadata": {
            "reward_signal": reward_signal,
            "current_turn": current_turn,
        },
    }


def _parse_answer(parsing: Any, model_output: str) -> Optional[Any]:
    if parsing.method == "regex" and parsing.pattern:
        match = re.search(parsing.pattern, model_output)
        return match.group(1) if match else None
    if parsing.method == "json_path" and parsing.json_path:
        try:
            data = json.loads(model_output)
            matches = jsonpath_parse(parsing.json_path).find(data)
            return matches[0].value if matches else None
        except Exception:
            return None
    return None


def _evaluate_reward_signal(
    creation: RewardCreationConfig,
    parsed_answer: Optional[Any],
    model_output: str,
    ground_truth: Any,
) -> Dict[str, Any]:
    method = creation.method
    found_answer = parsed_answer is not None
    raw_score: Optional[float] = None
    details: Dict[str, Any] = {}

    if method == "parsed_answer_rule":
        rule = creation.rule_type
        if rule == "exact_match":
            is_correct = (
                parsed_answer is not None
                and str(parsed_answer).strip() == str(ground_truth).strip()
            )
        elif rule == "regex_match":
            pattern = creation.regex_pattern or str(ground_truth)
            is_correct = (
                parsed_answer is not None
                and re.match(str(pattern), str(parsed_answer)) is not None
            )
        elif rule == "numeric_tolerance":
            tolerance = creation.numeric_tolerance or 0.01
            try:
                if parsed_answer is not None:
                    answer_num = float(parsed_answer)
                    gt_num = float(ground_truth)
                    is_correct = abs(answer_num - gt_num) <= tolerance
                else:
                    is_correct = False
            except (ValueError, TypeError):
                is_correct = False
        else:
            is_correct = False
    elif method == "json_path_rule":
        json_path = creation.json_path
        is_correct = False
        found_answer = False
        if json_path:
            try:
                data = json.loads(model_output)
                matches = jsonpath_parse(json_path).find(data)
                if matches:
                    value = matches[0].value
                    details["json_value"] = value
                    found_answer = True
                    success_values = creation.json_success_values or []
                    threshold = creation.json_threshold
                    if success_values:
                        is_correct = value in success_values
                    elif threshold is not None:
                        try:
                            raw_score = float(value)
                            is_correct = raw_score >= float(threshold)
                        except (TypeError, ValueError):
                            is_correct = False
            except Exception as exc:
                details["json_error"] = str(exc)
    elif method == "llm_verifier":
        found_answer = True
        judge = _call_llm_judge(creation.llm, model_output, ground_truth)
        is_correct = judge["is_correct"]
        raw_score = judge.get("score")
        details["judge_response"] = judge.get("response")
    else:
        is_correct = False

    return {
        "is_correct": is_correct,
        "found_answer": found_answer,
        "raw_score": raw_score,
        "details": details,
    }


def _apply_reward_scheme(
    scheme: RewardSchemeConfig,
    signal: Dict[str, Any],
) -> float:
    scheme_type = scheme.scheme

    if scheme_type == "dense" and signal.get("raw_score") is not None:
        return float(signal["raw_score"])

    if signal["is_correct"]:
        return scheme.correct_reward

    if not signal["found_answer"]:
        return scheme.format_error_reward or scheme.incorrect_reward

    if scheme.partial_reward is not None and scheme_type in {"partial", "dense"}:
        return scheme.partial_reward

    return scheme.incorrect_reward


def _determine_done(
    spec: EnvironmentSpec,
    is_correct: bool,
    current_turn: int,
) -> bool:
    if spec.env_type == "single_turn":
        return True
    if spec.done_condition == "always_single_step":
        return True
    if spec.done_condition == "max_turns_only":
        return current_turn >= spec.max_turns
    if spec.done_condition == "correct_or_max_turns":
        return current_turn >= spec.max_turns or is_correct
    return False


def _call_llm_judge(
    llm_config: Optional[LLMVerifierConfig],
    model_output: str,
    ground_truth: Any,
) -> Dict[str, Any]:
    if llm_config is None:
        return {"is_correct": False, "score": None, "response": "LLM config missing"}

    prompt = llm_config.prompt_template.format(
        model_output=model_output,
        ground_truth=ground_truth,
    )

    if llm_config.provider == "mock":
        score = 1.0 if str(ground_truth).strip() in model_output else 0.0
        is_correct = score >= llm_config.success_threshold
        return {
            "is_correct": is_correct,
            "score": score,
            "response": "mock heuristic",
        }

    if llm_config.provider == "openai":
        api_key = os.getenv(llm_config.api_key_env or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"LLM verifier requires API key set via {llm_config.api_key_env}"
            )
        base_url = (llm_config.api_base or "https://api.openai.com/v1").rstrip("/")
        payload = {
            "model": llm_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_output_tokens,
        }
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        score = _extract_score_from_response(content)

        if llm_config.response_format == "boolean":
            lowered = content.lower()
            is_correct = any(
                keyword.lower() in lowered for keyword in llm_config.success_keywords
            )
            score = 1.0 if is_correct else 0.0
        else:
            score = score if score is not None else 0.0
            is_correct = score >= llm_config.success_threshold

        return {"is_correct": is_correct, "score": score, "response": content}

    raise ValueError(f"Unsupported LLM provider '{llm_config.provider}'")


def _extract_score_from_response(content: str) -> Optional[float]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "score" in parsed:
            return float(parsed["score"])
    except Exception:
        pass

    match = re.search(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", content, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

