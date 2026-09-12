from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import httpx
from openai import APITimeoutError, AuthenticationError, PermissionDeniedError
import pytest
from pydantic import BaseModel, ValidationError

import layerread.analyzer as analyzer_module
from layerread.analysis_schema import AnalysisPayload
from layerread.analyzer import (
    AnalysisError,
    OpenAICompatibleClient,
    analyze_article,
    article_fingerprint,
    build_analysis_messages,
    parse_analysis,
)
from layerread.article import ArticleImage, process_article
from layerread.config import ModelSettings
from layerread.references import (
    normalize_reference_values,
    normalize_references,
    validate_references,
    valid_only,
)


def make_article():
    return process_article(
        "\n\n".join(
            [
                "人工智能工具可以缩短重复任务的处理时间，但工具本身并不保证工作质量。",
                "作者认为，真正的收益来自流程重构，而不是单纯追求生成速度。",
                "一个团队先用模型生成测试草稿，再由工程师审查边界条件和测试意图。",
                "这个案例说明自动化适合生成初稿，人类仍需承担验证与最终判断责任。",
                "文章没有提供大样本数据，因此案例是否能够推广到其他团队仍待验证。",
            ]
        ),
        reading_goal="判断是否适合团队采用",
        focus_area="实践",
    )


def cited(text: str, refs: list[str] | None = None) -> dict:
    return {"text": text, "paragraph_refs": refs or ["[P01]"]}


def criterion(score: int = 70, refs: list[str] | None = None) -> dict:
    return {
        "score": score,
        "explanation": "有一定信息价值，并有原文依据。",
        "paragraph_refs": refs or ["[P01]"],
    }


def valid_payload(refs: list[str] | None = None) -> dict:
    source_refs = refs or ["[P01]"]
    return {
        "reading_decision": {
            "recommendation": "建议重点阅读",
            "overall_value": criterion(78, source_refs),
            "information_density": criterion(72, source_refs),
            "evidence_quality": criterion(55, ["[P03]", "[P05]"]),
            "original_analysis": criterion(68, ["[P02]"]),
            "goal_relevance": criterion(88, ["[P02]", "[P04]"]),
            "promotion_likelihood": criterion(12, source_refs),
            "fluff_likelihood": criterion(20, source_refs),
            "confidence": criterion(82, ["[P01]", "[P05]"]),
            "recommended_approach": "重点阅读观点、案例与局限段落。",
        },
        "promotion_assessment": {
            "probability": 12,
            "potential_targets": [],
            "rationale": "没有明确产品或服务导向。",
            "paragraph_refs": source_refs,
            "valuable_parts": [cited("流程观点仍有价值", ["[P02]"])],
        },
        "fluff_assessment": {
            "effective_point_count": 3,
            "repetition": "重复较少。",
            "title_overstates_content": "未提供标题，无法判断。",
            "compressibility": 25,
            "paragraph_refs": ["[P01]", "[P02]"],
            "skippable_parts": [],
        },
        "core_analysis": {
            "core_question": "团队应如何使用 AI 工具并保留质量控制？",
            "one_sentence_conclusion": "AI 应增强判断，而不是替代验证。",
            "author_intended_view": "收益来自流程重构与人机分工。",
            "structure": [
                {
                    "order": 1,
                    "title": "提出问题",
                    "summary": "区分速度与质量。",
                    "paragraph_refs": ["[P01]", "[P02]"],
                },
                {
                    "order": 2,
                    "title": "案例与边界",
                    "summary": "用案例说明人机分工并指出证据局限。",
                    "paragraph_refs": ["[P03]", "[P04]", "[P05]"],
                },
            ],
            "knowledge_points": [
                {
                    "title": f"知识点 {index}",
                    "definition": "关于流程、人机分工或证据边界的概念。",
                    "plain_explanation": "工具先打草稿，人再检查关键问题。",
                    "example": "模型生成测试后由工程师审查。",
                    "applicability": "适用于结果可由人复核的任务。",
                    "common_misconceptions": ["速度提升等于质量提升"],
                    "paragraph_refs": [f"[P{index:02d}]"],
                }
                for index in range(1, 4)
            ],
            "claims": [
                {
                    "claim": "真正收益来自流程重构。",
                    "classification": "观点",
                    "evidence": "文章提供了单一团队案例。",
                    "evidence_type": "案例",
                    "evidence_strength": "中",
                    "logical_leaps": ["单一案例不能证明普遍有效"],
                    "paragraph_refs": ["[P02]", "[P03]"],
                }
            ],
            "limitations": [cited("缺少大样本数据", ["[P05]"])],
            "counterarguments": [cited("部分标准任务可以高度自动化", ["[P01]"])],
            "implicit_assumptions": [cited("团队具备人工复核能力", ["[P03]"])],
            "possible_counterexamples": [cited("规则稳定的任务可能无需逐项人工复核", ["[P04]"])],
            "applicability_boundaries": [cited("适用于可复核任务", ["[P04]"])],
            "items_to_verify": [cited("能否推广到其他团队", ["[P05]"])],
        },
        "explanations": [
            {
                "kind": "AI解释",
                "text": "这里的流程重构是重新分配机器与人的责任。",
                "paragraph_refs": ["[P02]", "[P04]"],
            },
            {
                "kind": "AI推断",
                "text": "团队需要先建立明确的复核标准。",
                "paragraph_refs": ["[P03]"],
            },
        ],
    }


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return next(self.outputs)


class ReferenceContainer(BaseModel):
    paragraph_refs: list[str]


class ReferenceBundle(BaseModel):
    sections: list[ReferenceContainer]


@pytest.fixture
def settings() -> ModelSettings:
    return ModelSettings(
        api_key="test-secret",
        base_url="https://models.example.com/v1",
        model="test-model",
    )


def test_schema_requires_at_least_three_complete_knowledge_points() -> None:
    payload = valid_payload()
    payload["core_analysis"]["knowledge_points"] = payload["core_analysis"]["knowledge_points"][:2]

    with pytest.raises(ValidationError):
        AnalysisPayload.model_validate(payload)


def test_schema_allows_no_forced_counterargument() -> None:
    payload = valid_payload()
    payload["core_analysis"]["counterarguments"] = []

    parsed = AnalysisPayload.model_validate(payload)

    assert parsed.core_analysis.counterarguments == []


def test_schema_bounds_model_output_size() -> None:
    payload = valid_payload()
    payload["core_analysis"]["knowledge_points"] *= 3

    with pytest.raises(ValidationError):
        AnalysisPayload.model_validate(payload)

    payload = valid_payload()
    payload["reading_decision"]["recommended_approach"] = "过长" * 301
    with pytest.raises(ValidationError):
        AnalysisPayload.model_validate(payload)


def test_structure_sections_allow_long_article_reference_spans() -> None:
    payload = valid_payload()
    payload["core_analysis"]["structure"][0]["paragraph_refs"] = [
        f"[P{index:02d}]" for index in range(1, 65)
    ]

    parsed = AnalysisPayload.model_validate(payload)

    assert len(parsed.core_analysis.structure[0].paragraph_refs) == 64

    payload["core_analysis"]["structure"][0]["paragraph_refs"].append("[P65]")
    with pytest.raises(ValidationError):
        AnalysisPayload.model_validate(payload)


def test_ai_inference_fields_may_leave_references_empty_instead_of_guessing() -> None:
    payload = valid_payload()
    payload["core_analysis"]["counterarguments"][0]["paragraph_refs"] = []
    payload["core_analysis"]["implicit_assumptions"][0]["paragraph_refs"] = []
    payload["core_analysis"]["possible_counterexamples"][0]["paragraph_refs"] = []

    parsed = AnalysisPayload.model_validate(payload)
    normalized, diagnostics = normalize_references(parsed, make_article())

    assert normalized.core_analysis.counterarguments[0].paragraph_refs == []
    assert diagnostics.invalid_mentions == 0


@pytest.mark.parametrize(
    ("mutate",),
    [
        (lambda payload: payload["reading_decision"]["overall_value"].update(score="78"),),
        (lambda payload: payload["core_analysis"].update(core_question="   "),),
        (lambda payload: payload.update(undocumented_field=True),),
    ],
)
def test_schema_rejects_coerced_blank_and_extra_values(mutate) -> None:
    payload = valid_payload()
    mutate(payload)

    with pytest.raises(ValidationError):
        AnalysisPayload.model_validate(payload)


def test_json_fence_is_accepted_but_schema_is_still_enforced() -> None:
    raw = f"```json\n{json.dumps(valid_payload(), ensure_ascii=False)}\n```"

    parsed = parse_analysis(raw)

    assert parsed.reading_decision.recommendation == "建议重点阅读"


def test_prompt_enumerates_only_references_available_in_current_article() -> None:
    article = make_article()
    messages = build_analysis_messages(article)
    prompt = messages[1]["content"]

    assert "合法段落编号仅限：[P01], [P02], [P03], [P04], [P05]" in prompt
    schema = analyzer_module._analysis_schema_with_allowed_references(article)

    enums: list[list[str]] = []

    def collect(value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "paragraph_refs":
                    enums.append(child["items"]["enum"])
                else:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(schema)
    assert enums
    assert all(enum == ["[P01]", "[P02]", "[P03]", "[P04]", "[P05]"] for enum in enums)


def test_multimodal_prompt_keeps_image_context_and_full_data_url() -> None:
    image = ArticleImage(
        image_id="IMG01",
        data_url="data:image/png;base64,ZmFrZQ==",
        mime_type="image/png",
        width=2400,
        height=1350,
        caption="AI 生成的产品路线 PPT",
        context_before="下面是一页路线图",
        context_after="作者随后解释实施节奏",
        original_width=2400,
        original_height=1350,
    )
    article = replace(
        make_article(),
        images=(image,),
        image_count_detected=1,
        media_dependency=True,
        media_completeness="images_included",
    )

    messages = build_analysis_messages(article)
    content = messages[1]["content"]

    assert isinstance(content, list)
    assert content[1]["text"].startswith("图片 IMG01")
    assert "产品路线 PPT" in content[1]["text"]
    assert content[2] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
    }


def test_images_can_be_excluded_for_text_only_provider() -> None:
    article = replace(
        make_article(),
        images=(
            ArticleImage(
                image_id="IMG01",
                data_url="data:image/png;base64,ZmFrZQ==",
                mime_type="image/png",
                width=320,
                height=180,
            ),
        ),
        image_count_detected=1,
        media_dependency=True,
        media_completeness="images_included",
    )

    messages = build_analysis_messages(article, include_images=False)

    assert isinstance(messages[1]["content"], str)
    assert "依赖未能传入当前模型的图片" in messages[1]["content"]


def test_reference_validator_identifies_nonexistent_paragraphs() -> None:
    article = make_article()
    payload = AnalysisPayload.model_validate(valid_payload(["[P99]"]))

    report = validate_references(payload, article)

    assert "[P99]" in report.invalid
    assert "[P03]" in report.valid
    assert valid_only(["[P01]", "[P99]"], article) == ["[P01]"]


@pytest.mark.parametrize(
    "raw_reference",
    ["[P1]", "P01", "[p01]", " ［Ｐ０１］ "],
)
def test_safe_single_reference_formats_are_auto_corrected(raw_reference: str) -> None:
    result = normalize_reference_values([raw_reference], make_article())

    assert result.valid == ("[P01]",)
    assert result.invalid == ()
    assert result.corrections


@pytest.mark.parametrize(
    ("raw_reference", "expected"),
    [
        ("[P01], [P02]", ("[P01]", "[P02]")),
        ("P01，P02", ("[P01]", "[P02]")),
        ("[P01、P02]", ("[P01]", "[P02]")),
        ("[P01][P02]", ("[P01]", "[P02]")),
        ("[P01] [P02]", ("[P01]", "[P02]")),
        ("P01/P02", ("[P01]", "[P02]")),
        ("P01 P02", ("[P01]", "[P02]")),
        ("[P01-P03]", ("[P01]", "[P02]", "[P03]")),
        ("[P01]-[P03]", ("[P01]", "[P02]", "[P03]")),
        ("P01—P03", ("[P01]", "[P02]", "[P03]")),
    ],
)
def test_safe_lists_and_ranges_are_expanded(
    raw_reference: str,
    expected: tuple[str, ...],
) -> None:
    result = normalize_reference_values([raw_reference], make_article())

    assert result.valid == expected
    assert result.invalid == ()


def test_reference_normalization_supports_p100_and_conservative_range_limit() -> None:
    article = process_article(
        "\n\n".join(
            f"第 {index} 段包含足够明确的测试内容，用于验证三位数段落编号。"
            for index in range(1, 101)
        )
    )

    assert normalize_reference_values(["P100"], article).valid == ("[P100]",)
    assert normalize_reference_values(["P99-P100"], article).valid == (
        "[P99]",
        "[P100]",
    )
    assert len(normalize_reference_values(["P01-P20"], article).valid) == 20
    oversized = normalize_reference_values(["P01-P21"], article)
    assert oversized.valid == ()
    assert oversized.invalid == ("P01-P21",)


@pytest.mark.parametrize(
    "raw_reference",
    [
        "见[P01]",
        "[P01",
        "P01]",
        "P1.5",
        "P-1",
        "P0",
        "📖",
        "   ",
    ],
)
def test_ambiguous_reference_text_is_not_guessed(raw_reference: str) -> None:
    result = normalize_reference_values([raw_reference], make_article())

    assert result.valid == ()
    assert result.invalid
    assert result.corrections == ()


def test_mixed_valid_and_nonexistent_list_keeps_only_real_source_as_valid() -> None:
    result = normalize_reference_values(["[P01],[P99]"], make_article())

    assert result.valid == ("[P01]",)
    assert result.invalid == ("[P99]",)
    assert result.corrections


def test_diagnostics_distinguish_repeated_error_from_unique_error_count() -> None:
    bundle = ReferenceBundle(
        sections=[
            ReferenceContainer(paragraph_refs=["[P99]"]),
            ReferenceContainer(paragraph_refs=["[P99]"]),
            ReferenceContainer(paragraph_refs=["[P99]"]),
        ]
    )

    _, diagnostics = normalize_references(bundle, make_article())

    assert diagnostics.invalid_references == ["[P99]"]
    assert diagnostics.invalid_mentions == 3
    assert diagnostics.nonexistent_mentions == 3
    assert diagnostics.affected_locations == 3
    assert diagnostics.zero_valid_locations == 3
    assert diagnostics.risk_level == "high"


def test_location_counts_are_grouped_by_statement_not_reference_item() -> None:
    bundle = ReferenceBundle(
        sections=[
            ReferenceContainer(paragraph_refs=["[P01]", "[P99]", "[P98]"]),
            ReferenceContainer(paragraph_refs=["[P99]", "[P98]"]),
        ]
    )

    _, diagnostics = normalize_references(bundle, make_article())

    assert diagnostics.invalid_mentions == 4
    assert diagnostics.affected_locations == 2
    assert diagnostics.zero_valid_locations == 1


def test_duplicate_mentions_inside_combined_reference_are_counted() -> None:
    bundle = ReferenceBundle(
        sections=[ReferenceContainer(paragraph_refs=["[P99],[P99]"])]
    )

    _, diagnostics = normalize_references(bundle, make_article())

    assert diagnostics.invalid_references == ["[P99]"]
    assert diagnostics.invalid_mentions == 2
    assert diagnostics.total_mentions == 2


def test_valid_analysis_uses_one_model_call(settings: ModelSettings) -> None:
    article = make_article()
    client = FakeClient([json.dumps(valid_payload(), ensure_ascii=False)])

    result = analyze_article(article, settings, client_factory=lambda _: client)

    assert result.validation_status == "valid"
    assert result.model_name == "test-model"
    assert result.article_id == article_fingerprint(
        article,
        "test-model",
        "https://models.example.com/v1",
        settings.supports_vision,
    )
    assert len(client.calls) == 1


def test_byok_vision_opt_in_controls_image_delivery(settings: ModelSettings) -> None:
    article = replace(
        make_article(),
        images=(
            ArticleImage(
                image_id="IMG01",
                data_url="data:image/png;base64,ZmFrZQ==",
                mime_type="image/png",
                width=320,
                height=180,
            ),
        ),
        image_count_detected=1,
        media_dependency=True,
        media_completeness="images_included",
    )
    text_client = FakeClient([json.dumps(valid_payload(), ensure_ascii=False)])
    vision_client = FakeClient([json.dumps(valid_payload(), ensure_ascii=False)])

    analyze_article(article, settings, client_factory=lambda _: text_client)
    analyze_article(
        article,
        replace(settings, provider_id="byok", supports_vision=True),
        client_factory=lambda _: vision_client,
    )

    assert isinstance(text_client.calls[0][1]["content"], str)
    assert isinstance(vision_client.calls[0][1]["content"], list)
    assert any(
        item.get("type") == "image_url"
        for item in vision_client.calls[0][1]["content"]
    )


def test_truncated_analysis_is_retried_once_with_compact_constraints(
    settings: ModelSettings,
) -> None:
    class LengthThenValidClient:
        def __init__(self) -> None:
            self.calls = []

        def complete(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                raise analyzer_module.AnalysisOutputTruncated('{"incomplete":')
            return json.dumps(valid_payload(), ensure_ascii=False)

    client = LengthThenValidClient()

    result = analyze_article(make_article(), settings, client_factory=lambda _: client)

    assert result.validation_status == "repaired"
    assert len(client.calls) == 2
    assert "上一次生成因输出过长被截断" in client.calls[1][-1]["content"]


def test_compact_length_retry_never_triggers_a_third_schema_repair(
    settings: ModelSettings,
) -> None:
    class LengthThenInvalidClient:
        def __init__(self) -> None:
            self.calls = []

        def complete(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                raise analyzer_module.AnalysisOutputTruncated('{"incomplete":')
            return '{"still":"invalid"}'

    client = LengthThenInvalidClient()

    with pytest.raises(AnalysisError, match="自动精简重试"):
        analyze_article(make_article(), settings, client_factory=lambda _: client)

    assert len(client.calls) == 2


def test_format_only_reference_error_is_fixed_without_second_model_call(
    settings: ModelSettings,
) -> None:
    article = make_article()
    client = FakeClient([json.dumps(valid_payload(["[P1]"]), ensure_ascii=False)])

    result = analyze_article(article, settings, client_factory=lambda _: client)

    assert result.validation_status == "valid_with_auto_corrections"
    assert result.reading_decision.overall_value.paragraph_refs == ["[P01]"]
    assert result.reference_diagnostics.corrected_valid_mentions > 0
    assert result.reference_diagnostics.invalid_mentions == 0
    assert any(item.statement for item in result.reference_diagnostics.items)
    assert len(client.calls) == 1


def test_invalid_json_is_repaired_exactly_once(settings: ModelSettings) -> None:
    client = FakeClient(
        [
            "这不是 JSON",
            json.dumps(valid_payload(), ensure_ascii=False),
        ]
    )

    result = analyze_article(make_article(), settings, client_factory=lambda _: client)

    assert result.validation_status == "repaired"
    assert len(client.calls) == 2
    assert "未通过校验" in client.calls[1][-1]["content"]


def test_schema_repair_receives_safe_specific_field_errors(
    settings: ModelSettings,
) -> None:
    invalid = valid_payload()
    invalid["reading_decision"]["overall_value"]["score"] = "78"
    invalid["reading_decision"]["overall_value"]["paragraph_refs"] = [
        "[P01]"
    ] * 13
    invalid["core_analysis"]["knowledge_points"] = []
    invalid["undocumented_field"] = "must-not-appear-in-repair-summary"
    client = FakeClient(
        [
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(valid_payload(), ensure_ascii=False),
        ]
    )

    result = analyze_article(make_article(), settings, client_factory=lambda _: client)

    repair_prompt = client.calls[1][-1]["content"]
    assert result.validation_status == "repaired"
    assert "reading_decision.overall_value.score" in repair_prompt
    assert "必须是整数" in repair_prompt
    assert "reading_decision.overall_value.paragraph_refs" in repair_prompt
    assert "数组项目过多（最多 12 项）" in repair_prompt
    assert "core_analysis.knowledge_points" in repair_prompt
    assert "数组项目过少" in repair_prompt
    assert "undocumented_field" in repair_prompt
    assert "额外字段" in repair_prompt
    assert "must-not-appear-in-repair-summary" not in repair_prompt


def test_known_qwen_schema_variants_are_normalized_without_another_call(
    settings: ModelSettings,
) -> None:
    variant = valid_payload()
    variant["reading_decision"]["paragraph_refs"] = ["[P01]"]
    variant["fluff_assessment"]["title_overstates_content"] = False
    variant["core_analysis"]["counterarguments"] = []
    raw = json.dumps(variant, ensure_ascii=False)
    client = FakeClient([raw])

    with pytest.raises(AnalysisError):
        parse_analysis(raw)

    result = analyze_article(make_article(), settings, client_factory=lambda _: client)

    assert len(client.calls) == 1
    assert result.validation_status == "valid_with_auto_corrections"
    assert result.fluff_assessment.title_overstates_content == "否"
    assert result.core_analysis.counterarguments == []
    assert not hasattr(result.reading_decision, "paragraph_refs")


def test_invalid_json_after_repair_has_clear_error(settings: ModelSettings) -> None:
    client = FakeClient(["无效", "仍然无效"])

    with pytest.raises(AnalysisError, match="一次自动修复"):
        analyze_article(make_article(), settings, client_factory=lambda _: client)

    assert len(client.calls) == 2


def test_schema_failure_exposes_only_bounded_safe_diagnostic(
    settings: ModelSettings,
) -> None:
    invalid = valid_payload()
    invalid["reading_decision"]["overall_value"]["score"] = "78"
    invalid["private_model_text"] = "sensitive-value-must-not-be-shown"
    client = FakeClient(
        [
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(invalid, ensure_ascii=False),
        ]
    )

    with pytest.raises(AnalysisError) as caught:
        analyze_article(make_article(), settings, client_factory=lambda _: client)

    message = str(caught.value)
    assert "安全诊断" in message
    assert "reading_decision.overall_value.score" in message
    assert "private_model_text" in message
    assert "sensitive-value-must-not-be-shown" not in message
    assert len(message) < 700


def test_invalid_reference_triggers_one_repair(settings: ModelSettings) -> None:
    invalid = valid_payload(["[P99]"])
    client = FakeClient(
        [
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(valid_payload(), ensure_ascii=False),
        ]
    )

    result = analyze_article(make_article(), settings, client_factory=lambda _: client)

    assert result.validation_status == "repaired"
    assert result.invalid_references == []
    assert len(client.calls) == 2


def test_remaining_invalid_reference_is_recorded_not_treated_as_valid(
    settings: ModelSettings,
) -> None:
    invalid_json = json.dumps(valid_payload(["[P99]"]), ensure_ascii=False)
    client = FakeClient([invalid_json, invalid_json])

    result = analyze_article(make_article(), settings, client_factory=lambda _: client)

    assert result.validation_status == "valid_with_invalid_references"
    assert result.invalid_references == ["[P99]"]
    assert len(client.calls) == 2


def test_fingerprint_changes_with_model_or_reading_preference() -> None:
    article = make_article()
    other = process_article(
        article.raw_text,
        reading_goal="只关心商业价值",
        focus_area="商业",
    )

    assert article_fingerprint(article, "model-a") == article_fingerprint(article, "model-a")
    assert article_fingerprint(article, "model-a") != article_fingerprint(article, "model-b")
    assert article_fingerprint(article, "model-a") != article_fingerprint(other, "model-a")
    assert article_fingerprint(article, "model-a", "https://one.example/v1") != article_fingerprint(
        article,
        "model-a",
        "https://two.example/v1",
    )
    assert article_fingerprint(article, "model-a", supports_vision=False) != article_fingerprint(
        article,
        "model-a",
        supports_vision=True,
    )


def test_http_error_is_actionable_without_exposing_api_key(
    settings: ModelSettings,
    monkeypatch,
) -> None:
    request = httpx.Request("POST", "https://models.example.com/v1/chat/completions")
    response = httpx.Response(401, request=request, json={"error": {}})
    error = AuthenticationError("Unauthorized", response=response, body={"error": {}})

    class RejectingClient:
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(error))
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        analyzer_module,
        "_create_sdk_client",
        lambda *_args, **_kwargs: RejectingClient(),
    )

    with pytest.raises(AnalysisError) as captured:
        OpenAICompatibleClient(settings).complete(
            [{"role": "user", "content": "返回 JSON"}]
        )

    assert "HTTP 401" in str(captured.value)
    assert "test-secret" not in str(captured.value)


def test_aliyun_client_uses_openai_compatible_chat_completions(
    monkeypatch,
) -> None:
    captured = {}

    class CompatibleClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            captured["payload"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content='{"ok": true}'),
                    )
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def create_client(settings, route, timeout):
        captured["base_url"] = route.base_url
        captured["api_key"] = settings.api_key
        captured["timeout"] = timeout
        return CompatibleClient()

    monkeypatch.setattr(analyzer_module, "_create_sdk_client", create_client)
    aliyun_settings = ModelSettings(
        api_key="dashscope-secret",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-plus",
        provider_id="aliyun",
        provider_name="阿里云百炼",
    )

    content = OpenAICompatibleClient(aliyun_settings).complete(
        [{"role": "user", "content": "返回 JSON"}]
    )

    assert content == '{"ok": true}'
    assert captured["base_url"].endswith("/compatible-mode/v1")
    assert captured["api_key"] == "dashscope-secret"
    assert captured["payload"]["model"] == "qwen3.7-plus"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["extra_body"] == {"enable_thinking": False}
    assert captured["payload"]["max_completion_tokens"] == 32768
    assert "max_tokens" not in captured["payload"]

    OpenAICompatibleClient(aliyun_settings).complete(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "分析图片并返回 JSON"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                    },
                ],
            }
        ]
    )
    assert captured["payload"]["extra_body"] == {
        "enable_thinking": False,
        "vl_high_resolution_images": True,
    }


def test_official_deepseek_uses_same_output_budget_without_thinking(
    monkeypatch,
) -> None:
    captured = {}

    class CompatibleClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            captured["payload"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content='{"ok": true}'),
                    )
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        analyzer_module,
        "_create_sdk_client",
        lambda *_args, **_kwargs: CompatibleClient(),
    )
    deepseek_settings = ModelSettings(
        api_key="deepseek-secret",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
    )

    content = OpenAICompatibleClient(deepseek_settings).complete(
        [{"role": "user", "content": "返回 JSON"}]
    )

    assert content == '{"ok": true}'
    assert captured["payload"]["max_tokens"] == 32768
    assert "max_completion_tokens" not in captured["payload"]
    assert captured["payload"]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_aliyun_connect_timeout_falls_back_to_same_region_shared_endpoint(
    monkeypatch,
) -> None:
    attempts: list[str] = []
    request = httpx.Request("POST", "https://workspace.example/chat/completions")
    connect_timeout = APITimeoutError(request)
    connect_timeout.__cause__ = httpx.ConnectTimeout("TLS timeout", request=request)

    class Client:
        def __init__(self, route):
            self.route = route
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **_kwargs):
            attempts.append(self.route.base_url)
            if "workspace.cn-beijing" in self.route.base_url:
                raise connect_timeout
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content='{"ok": true}'),
                    )
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(analyzer_module, "_environment_proxy_url", lambda: "")
    monkeypatch.setattr(
        analyzer_module,
        "_create_sdk_client",
        lambda _settings, route, _timeout: Client(route),
    )
    aliyun_settings = ModelSettings(
        api_key="sk-ws-secret",
        base_url="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-plus",
        provider_id="aliyun",
        provider_name="阿里云百炼",
        fallback_base_urls=(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )

    content = OpenAICompatibleClient(aliyun_settings).complete(
        [{"role": "user", "content": "返回 JSON"}]
    )

    assert content == '{"ok": true}'
    assert attempts == [
        "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ]


def test_read_timeout_is_not_retried_to_avoid_duplicate_billing(monkeypatch) -> None:
    request = httpx.Request("POST", "https://workspace.example/chat/completions")
    read_timeout = APITimeoutError(request)
    read_timeout.__cause__ = httpx.ReadTimeout("response timeout", request=request)
    calls = 0

    class Client:
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(read_timeout))
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def create_client(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return Client()

    monkeypatch.setattr(analyzer_module, "_create_sdk_client", create_client)
    aliyun_settings = ModelSettings(
        api_key="sk-ws-secret",
        base_url="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-plus",
        provider_id="aliyun",
        provider_name="阿里云百炼",
        fallback_base_urls=(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )

    with pytest.raises(AnalysisError, match="等待模型生成超过 300 秒") as captured:
        OpenAICompatibleClient(aliyun_settings).complete(
            [{"role": "user", "content": "返回 JSON"}]
        )

    assert calls == 1
    assert "可能已在供应商侧产生用量" in str(captured.value)
    assert "并不等同于 HTTPS 连接失败" in str(captured.value)


def test_aliyun_plan_auth_error_explains_backend_restriction(
    monkeypatch,
) -> None:
    body = {
        "error": {
            "code": "invalid_api_key",
            "message": "Incorrect API key provided.",
        },
        "request_id": "request-safe-id",
    }
    request = httpx.Request(
        "POST",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    response = httpx.Response(401, request=request, json=body)
    error = AuthenticationError("Unauthorized", response=response, body=body)

    class RejectingClient:
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(error))
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        analyzer_module,
        "_create_sdk_client",
        lambda *_args, **_kwargs: RejectingClient(),
    )
    aliyun_settings = ModelSettings(
        api_key="sk-sp-private-secret",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.8-max-preview",
        provider_id="aliyun",
        provider_name="阿里云百炼",
    )

    with pytest.raises(AnalysisError) as captured:
        OpenAICompatibleClient(aliyun_settings).complete(
            [{"role": "user", "content": "返回 JSON"}]
        )

    message = str(captured.value)
    assert "HTTP 401" in message
    assert "sk-sp-" in message
    assert "应用后端" in message
    assert "按量付费" in message
    assert "private-secret" not in message
    assert "invalid_api_key" in message
    assert "request-safe-id" in message


def test_aliyun_free_tier_hard_stop_has_safe_actionable_message(
    monkeypatch,
) -> None:
    body = {
        "error": {
            "code": "AllocationQuota.FreeTierOnly",
            "message": "Free tier quota exhausted.",
        },
        "request_id": "request-free-tier-stop",
    }
    request = httpx.Request(
        "POST",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    response = httpx.Response(403, request=request, json=body)
    error = PermissionDeniedError("Forbidden", response=response, body=body)

    class RejectingClient:
        chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: (_ for _ in ()).throw(error)
            )
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        analyzer_module,
        "_create_sdk_client",
        lambda *_args, **_kwargs: RejectingClient(),
    )
    aliyun_settings = ModelSettings(
        api_key="sk-ws-private-secret",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.6-flash",
        provider_id="aliyun",
        provider_name="阿里云百炼",
    )

    with pytest.raises(AnalysisError) as captured:
        OpenAICompatibleClient(aliyun_settings).complete(
            [{"role": "user", "content": "返回 JSON"}]
        )

    message = str(captured.value)
    assert "免费额度已用完" in message
    assert "硬停损" in message
    assert "阻止继续计费" in message
    assert "private-secret" not in message
    assert "Free tier quota exhausted" not in message
    assert "AllocationQuota.FreeTierOnly" in message
    assert "request-free-tier-stop" in message


def test_truncated_provider_response_is_not_repaired_as_bad_json(
    settings: ModelSettings,
    monkeypatch,
) -> None:
    class TruncatedClient:
        chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="length",
                            message=SimpleNamespace(content='{"incomplete":'),
                        )
                    ]
                )
            )
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        analyzer_module,
        "_create_sdk_client",
        lambda *_args, **_kwargs: TruncatedClient(),
    )

    with pytest.raises(AnalysisError, match="长度上限"):
        OpenAICompatibleClient(settings).complete(
            [{"role": "user", "content": "返回 JSON"}]
        )
