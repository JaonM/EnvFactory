"""Deterministic quality scoring for generated Agentic-RL task artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DOMAIN_MARKER_GROUPS: dict[str, tuple[str, ...]] = {
    "city": ("城市", "候选城市", "气候", "降水", "博物馆", "地铁", "行政区"),
    "language": ("日语", "阿依努语", "多式综合语", "词汇", "语法", "语言学"),
    "clothing": ("服装", "衣物", "面料", "纤维", "手表", "表带", "鞋", "尺码"),
    "food": ("菜", "菜系", "食材", "烹饪", "餐厅", "营养", "火锅"),
    "device": ("设备", "服务器", "日志", "故障", "传感器", "网络", "路由器"),
    "commerce": ("商品", "库存", "采购", "订单", "价格", "供应", "预算"),
    "chemistry": ("化学", "溶剂", "树脂", "化合物", "甲醇", "乙醇", "丙酮"),
    "medical": ("医疗", "诊断", "症状", "用药", "疾病", "疼痛"),
    "legal": ("法律", "法规", "合规", "政策", "税务", "证券"),
    "event": ("活动", "签到", "日程", "会议", "读书会", "参与者"),
    "wildlife": ("动物", "藏羚羊", "雪豹", "野生", "自然观察"),
    "travel": ("候选地点", "目的地", "车程", "门票", "出行", "景点"),
}

CONTRADICTORY_DOMAIN_PAIRS = {
    frozenset(("language", "city")),
    frozenset(("language", "commerce")),
    frozenset(("language", "device")),
    frozenset(("wildlife", "travel")),
    frozenset(("food", "device")),
    frozenset(("clothing", "medical")),
}

HIGH_VALUE_INTENTS = {
    "compare", "recommend", "decide", "diagnose", "troubleshoot", "monitor",
    "plan", "audit", "validate", "execute", "schedule", "simulate",
}


@dataclass(frozen=True)
class QualityDimension:
    score: float
    maximum: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TaskQualityReport:
    task_id: str
    path: str
    score: float
    passed: bool
    tier: str
    dimensions: dict[str, QualityDimension]
    findings: tuple[str, ...]
    training_category: str = "multi_step_agentic"
    agentic_level: int = 2
    tool_policy_target: str = "follow_dependency_chain"
    eligible: bool = True
    eligibility_failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dimensions"] = {
            name: asdict(dimension) for name, dimension in self.dimensions.items()
        }
        return value


def _objects(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def _dimension(maximum: float, deductions: Iterable[tuple[float, str]]) -> QualityDimension:
    items = list(deductions)
    return QualityDimension(
        score=round(max(0.0, maximum - sum(amount for amount, _ in items)), 2),
        maximum=maximum,
        reasons=tuple(reason for _, reason in items),
    )


def _domain_labels(value: Any) -> set[str]:
    text = _text(value)
    return {
        label for label, markers in DOMAIN_MARKER_GROUPS.items()
        if any(marker in text for marker in markers)
    }


def _conflicting_domains(task_domains: set[str], candidate_domains: set[str]) -> set[str]:
    return {
        candidate for candidate in candidate_domains - task_domains
        if any(frozenset((task_domain, candidate)) in CONTRADICTORY_DOMAIN_PAIRS for task_domain in task_domains)
    }


def score_task(task: dict[str, Any], *, task_id: str = "task", path: str = "", min_score: float = 8.0) -> TaskQualityReport:
    """Score one task artifact on a stable 0-10 training-quality rubric."""
    task_text = str(task.get("task", "")).strip()
    requirements = task.get("requirements") if isinstance(task.get("requirements"), dict) else {}
    semantic_requirements = {
        key: value for key, value in requirements.items()
        if key not in {"runtime_interface", "media_truth_mode"}
    }
    mode = task.get("environment_plan", {}).get("mode")
    actions = _objects(task.get("actions"))
    tools = _objects(task.get("tools"))
    noise = _objects(task.get("noise_tools"))
    metrics = _objects(task.get("metrics"))
    implementations = _objects(task.get("metric_implementations"))
    contract = task.get("acceptance_contract") if isinstance(task.get("acceptance_contract"), dict) else {}
    readiness = task.get("task_readiness") if isinstance(task.get("task_readiness"), dict) else {}
    declared_training_category = task.get("training_category")
    category_profiles = {
        "direct_response": (0, "do_not_call"),
        "simple_agentic": (1, "call_required_business_tool"),
        "multi_step_agentic": (2, "follow_dependency_chain"),
    }
    noise_names = {str(item.get("name")) for item in noise}
    tool_names = {
        str(tool.get("function", {}).get("name"))
        for tool in tools if isinstance(tool.get("function"), dict)
    }
    business_names = tool_names - noise_names
    if declared_training_category in category_profiles:
        training_category = str(declared_training_category)
    elif not business_names:
        training_category = "direct_response"
    elif len(business_names) == 1:
        training_category = "simple_agentic"
    else:
        training_category = "multi_step_agentic"
    agentic_level, tool_policy_target = category_profiles[training_category]
    corpus = _text({"task": task_text, "requirements": semantic_requirements})
    statement_domains = _domain_labels(task_text)
    requirement_domains = _domain_labels(semantic_requirements)
    task_domains = statement_domains | requirement_domains
    requirement_conflicts = _conflicting_domains(statement_domains, requirement_domains)
    action_domains = _domain_labels(actions)
    metric_domains = _domain_labels(metrics)
    business_tools = [
        tool for tool in tools
        if str(tool.get("function", {}).get("name")) in business_names
    ]
    tool_domains = _domain_labels(business_tools)

    contract_deductions: list[tuple[float, str]] = []
    if len(task_text) < 16:
        contract_deductions.append((0.6, "任务描述过短，目标或边界可能不完整"))
    if not requirements:
        contract_deductions.append((0.5, "缺少 requirements"))
    if not actions:
        contract_deductions.append((0.7, "缺少可执行动作分解"))
    if task.get("complexity") not in {"simple", "standard", "complex"}:
        contract_deductions.append((0.4, "complexity 无效"))
    if any(marker in corpus for marker in ("待补充", "自行假设", "相关信息等", "视情况")):
        contract_deductions.append((0.4, "任务契约包含模糊或未决输入"))
    if requirement_conflicts:
        contract_deductions.append((1.2, "requirements 与任务描述发生领域冲突：" + ", ".join(sorted(requirement_conflicts))))

    challenge_deductions: list[tuple[float, str]] = []
    if task.get("complexity") == "simple" and training_category == "multi_step_agentic":
        challenge_deductions.append((0.35, "任务标记为 simple，长程决策密度有限"))
    if len(actions) < 3 and training_category == "multi_step_agentic":
        challenge_deductions.append((0.65, "少于 3 个语义动作，Agentic 轨迹过短"))
    if not business_names and training_category != "direct_response":
        challenge_deductions.append((0.45, "没有业务工具，只能训练拒绝噪声工具"))
    if len(metrics) < 2:
        challenge_deductions.append((0.35, "奖励信号维度不足"))

    alignment_deductions: list[tuple[float, str]] = []
    if mode not in {"stateless", "reference_data", "stateful", "external_capability"}:
        alignment_deductions.append((0.8, "环境模式缺失或无效"))
    explicit_user_only = any(marker in corpus for marker in ("仅依赖用户", "仅使用用户", "只依赖用户", "只使用用户"))
    complete_input_supplied = any(
        re.search(pattern, corpus) is not None
        for pattern in (
            r"基于用户提供的.{0,8}(?:规格|数据|资料|文本)",
            r"基于以下提供的.{0,8}(?:规格|数据|资料|文本)",
            r"从用户提供的.{0,8}(?:列表|清单|文本|数据)中",
        )
    )
    if mode in {"reference_data", "stateful"} and (explicit_user_only or complete_input_supplied):
        alignment_deductions.append((0.8, "仅依赖用户输入的任务被过度设计为数据环境"))
    if mode in {"reference_data", "stateful"} and not business_names:
        alignment_deductions.append((0.8, "数据环境没有业务数据访问工具"))
    if mode == "external_capability" and not business_names:
        alignment_deductions.append((0.8, "外部能力任务没有业务查询工具"))
    if len(noise_names) != len(noise) or not noise_names <= tool_names:
        alignment_deductions.append((0.6, "噪声工具元数据与工具定义不一致"))
    action_text = _text(actions)
    for marker in ("生活成本", "预算排序", "评分排序"):
        if marker in action_text and marker not in corpus:
            alignment_deductions.append((0.5, f"动作引入任务外约束：{marker}"))
            break
    # Domain comparison is only authoritative when the task itself maps to at
    # least one known domain. Otherwise a generic word such as "商品" in a
    # legitimate tool description would create a false cross-domain verdict.
    imported_action_domains = _conflicting_domains(task_domains, action_domains)
    if imported_action_domains:
        alignment_deductions.append((1.2, "动作引入任务外领域：" + ", ".join(sorted(imported_action_domains))))
    imported_metric_domains = _conflicting_domains(task_domains, metric_domains)
    if imported_metric_domains:
        alignment_deductions.append((0.8, "奖励指标评估了任务外领域：" + ", ".join(sorted(imported_metric_domains))))
    imported_tool_domains = _conflicting_domains(task_domains, tool_domains)
    if imported_tool_domains:
        alignment_deductions.append((0.8, "业务工具属于任务外领域：" + ", ".join(sorted(imported_tool_domains))))

    reward_deductions: list[tuple[float, str]] = []
    categories = {str(metric.get("category")) for metric in metrics}
    if "outcome" not in categories:
        reward_deductions.append((0.8, "缺少 outcome 指标"))
    if noise and "penalty" not in categories:
        reward_deductions.append((0.5, "存在噪声工具但缺少轨迹惩罚指标"))
    if any(not isinstance(metric.get("evaluator"), dict) for metric in metrics):
        reward_deductions.append((0.6, "指标缺少可执行 evaluator"))
    rule_metric_ids = {
        str(metric.get("id")) for metric in metrics if metric.get("type") == "rule-based"
    }
    implemented_ids = {str(item.get("metric_id")) for item in implementations}
    if rule_metric_ids - implemented_ids:
        reward_deductions.append((0.5, "部分 rule-based 指标缺少声明式实现"))
    formula = task.get("reward_formula")
    if not isinstance(formula, dict) or formula.get("score_range") != [-1, 1]:
        reward_deductions.append((0.6, "奖励公式或范围不完整"))

    acceptance_deductions: list[tuple[float, str]] = []
    scenarios = _objects(contract.get("executable_scenarios"))
    scenario_kinds = {str(item.get("kind")) for item in scenarios}
    if not {"goal_success", "goal_failure"} <= scenario_kinds:
        acceptance_deductions.append((0.8, "缺少成功/失败可执行场景"))
    if noise and "noise_selection" not in scenario_kinds:
        acceptance_deductions.append((0.5, "缺少噪声选择场景"))
    if not contract.get("mutation_tests"):
        acceptance_deductions.append((0.4, "缺少 mutation tests"))
    if readiness.get("ready") is not True:
        acceptance_deductions.append((0.5, "任务未声明 task_readiness.ready=true"))
    success_scenarios = [item for item in scenarios if item.get("kind") == "goal_success"]
    success_business_calls = [
        step for scenario in success_scenarios for step in _objects(scenario.get("steps"))
        if step.get("operation") == "tool_call" and step.get("tool_name") in business_names
    ]
    dependency_edge = False
    for scenario in success_scenarios:
        captures: set[str] = set()
        for step in _objects(scenario.get("steps")):
            if step.get("operation") == "tool_call":
                argument_text = _text(step.get("arguments", {}))
                if any(f'"$ref": "{name.lower()}"' in argument_text for name in captures):
                    dependency_edge = True
            capture = step.get("capture")
            if isinstance(capture, dict):
                captures.update(str(name) for name in capture)
    if training_category == "direct_response":
        if business_names or success_business_calls:
            alignment_deductions.append((1.0, "direct_response 不应依赖业务工具"))
        if mode != "stateless":
            alignment_deductions.append((0.8, "direct_response 应使用 stateless 环境"))
    elif training_category == "multi_step_agentic" and len(success_business_calls) < 2:
        acceptance_deductions.append((1.2, "multi_step_agentic 成功轨迹至少需要两个业务工具调用"))
    elif training_category == "multi_step_agentic" and not dependency_edge:
        acceptance_deductions.append((1.0, "multi_step_agentic 缺少 capture/$ref 工具数据依赖"))
    elif training_category == "simple_agentic" and not success_business_calls:
        acceptance_deductions.append((1.2, f"{training_category} 缺少必要业务工具调用"))
    bad_success_fixture = False
    for scenario in success_scenarios:
        for step in _objects(scenario.get("steps")):
            if step.get("operation") != "tool_call":
                continue
            arguments = step.get("arguments")
            argument_text = _text(arguments)
            schema = next((item.get("function", {}).get("parameters", {}) for item in _objects(task.get("tools"))
                           if item.get("function", {}).get("name") == step.get("tool_name")), {})
            if not isinstance(arguments, dict) or not set(schema.get("required", [])) <= set(arguments):
                bad_success_fixture = True
            if any(marker in argument_text for marker in (
                "fixture-value", "placeholder", "sample-value", "test-value"
            )):
                bad_success_fixture = True
            if isinstance(arguments, dict) and any(value in ([], {}) for value in arguments.values()):
                bad_success_fixture = True
    if bad_success_fixture:
        acceptance_deductions.append((1.2, "成功轨迹包含空参数或占位业务值"))
    if not success_business_calls and training_category != "direct_response":
        acceptance_deductions.append((1.2, "成功轨迹未执行任何业务工具，无法验证 Agentic 决策"))
    warnings = readiness.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        acceptance_deductions.append((min(0.5, 0.1 * len(warnings)), "task_readiness 含警告"))

    dimensions = {
        "task_contract": _dimension(2.0, contract_deductions),
        "agentic_challenge": _dimension(2.0, challenge_deductions),
        "environment_tool_alignment": _dimension(2.0, alignment_deductions),
        "reward_evaluability": _dimension(2.0, reward_deductions),
        "acceptance_readiness": _dimension(2.0, acceptance_deductions),
    }
    score = round(sum(item.score for item in dimensions.values()), 2)
    findings_list = [reason for item in dimensions.values() for reason in item.reasons]
    # Eligibility is a hard gate, separate from the descriptive quality score.
    # An invalid training sample must never be represented as merely "7.8".
    category_gate_failed = (
        (training_category == "direct_response" and bool(business_names or success_business_calls or mode != "stateless"))
        or (training_category == "simple_agentic" and not success_business_calls)
        or (training_category == "multi_step_agentic" and (len(success_business_calls) < 2 or not dependency_edge))
    )
    eligibility_failures: list[str] = []
    if category_gate_failed:
        eligibility_failures.append(f"任务不满足 {training_category} 路由契约")
    if any("过度设计为数据环境" in item for item in findings_list):
        eligibility_failures.append("环境模式与任务输入边界冲突")
    if bad_success_fixture:
        eligibility_failures.append("成功轨迹不能作为真实 Agentic 训练正样本")
    if requirement_conflicts or imported_action_domains or imported_metric_domains or imported_tool_domains:
        eligibility_failures.append("任务、动作、工具或奖励发生跨领域语义漂移")
    task_spec = task.get("task_spec")
    if not isinstance(task_spec, dict):
        eligibility_failures.append("缺少可编译的 task_spec IR")
    else:
        try:
            from .task_spec import validate_task_spec
            validate_task_spec(task_spec)
        except (ValueError, TypeError) as exc:
            eligibility_failures.append(f"task_spec IR 无效：{exc}")
    if readiness.get("ready") is not True:
        eligibility_failures.append("task_readiness 未通过")
    eligible = not eligibility_failures
    findings_list.extend(f"训练资格失败：{reason}" for reason in eligibility_failures)
    findings = tuple(findings_list)
    passed = eligible and score >= min_score
    high_value = (
        passed
        and score >= 9.0
        and not category_gate_failed
        and not bad_success_fixture
        and not requirement_conflicts
        and not imported_action_domains
        and not imported_metric_domains
        and not imported_tool_domains
    )
    tier = "high_value" if high_value else ("usable" if passed else "rejected")
    return TaskQualityReport(
        task_id, path, score, passed, tier, dimensions, findings,
        training_category, agentic_level, tool_policy_target, eligible,
        tuple(eligibility_failures),
    )


def score_file(path: Path, *, min_score: float = 8.0) -> TaskQualityReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: task artifact must be a JSON object")
    task_id = path.parent.name if path.name == "task.json" else path.stem
    report = score_task(data, task_id=task_id, path=str(path), min_score=min_score)
    manifest = data.get("artifacts", {}).get("data_manifest", {})
    declared_root = Path(str(manifest.get("root", ".")))
    root = declared_root if declared_root.is_absolute() else path.parent / declared_root
    if not root.is_dir() and declared_root.is_dir():
        root = declared_root
    tables = []
    for table in _objects(manifest.get("tables")):
        schema_path = root / table["schema_file"]
        rows_path = root / table["rows_file"]
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        tables.append({**schema, "table_name": table["table_name"], "rows": rows})
    if isinstance(data.get("task_spec"), dict):
        from .task_spec import validate_task_spec
        try:
            validate_task_spec(data["task_spec"], data_tables=tables)
        except ValueError as exc:
            finding = f"任务实际数据不满足 TaskSpec：{exc}"
            return TaskQualityReport(
                report.task_id, report.path, report.score, False, "rejected",
                report.dimensions, report.findings + (finding,), report.training_category,
                report.agentic_level, report.tool_policy_target, False,
                report.eligibility_failures + (finding,),
            )
    # New write values and runtime $refs need not occur in the initial rows.
    # Their semantics are verified by execution, not field-name membership.
    return report


def discover_task_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    def key(item: Path) -> tuple[int, str]:
        match = re.fullmatch(r"task-(\d+)", item.parent.name)
        return (int(match.group(1)), item.parent.name) if match else (10**18, item.parent.name)

    return sorted(root.glob("task-*/task.json"), key=key)


def error_report(path: Path, error: Exception, *, min_score: float = 8.0) -> TaskQualityReport:
    reason = f"任务文件无法解析或评分：{error}"
    return TaskQualityReport(
        task_id=path.parent.name if path.name == "task.json" else path.stem,
        path=str(path),
        score=0.0,
        passed=False,
        tier="rejected",
        dimensions={"artifact_integrity": QualityDimension(0.0, 10.0, (reason,))},
        findings=(reason,),
    )
