"""Description request construction and validation, without network calls."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

DESCRIPTION_FIELDS: Final[tuple[str, str, str]] = ("appearance", "functionality", "environment")
DESCRIPTION_VALIDATION_STAGES: Final[tuple[str, ...]] = ("unknown", "response_rows", "prompt_build")
DESCRIPTION_VALIDATION_RULE_CODES: Final[tuple[str, ...]] = (
    "unspecified",
    "row_count",
    "class_order",
    "field_type",
    "empty",
    "control",
    "word_count",
    "non_english",
    "duplicate",
    "token_counter",
    "token_budget",
    "field_missing",
)
DESCRIPTION_VALIDATION_FIELDS: Final[tuple[str, ...]] = ("class_name", *DESCRIPTION_FIELDS)
MAX_VALIDATION_ROW_INDEX: Final[int] = 1000000
MIN_DESCRIPTION_WORDS: Final[int] = 8
MAX_DESCRIPTION_WORDS: Final[int] = 18
_ENGLISH_WORD_RE: Final[re.Pattern[str]] = re.compile("^[A-Za-z]+(?:['-][A-Za-z]+)*$")
_CONTROL_RE: Final[re.Pattern[str]] = re.compile("[\\x00-\\x1f\\x7f]")
_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    "^\\s*```(?:json)?\\s*(.*?)\\s*```\\s*$", re.IGNORECASE | re.DOTALL
)


class DescriptionGenerationError(RuntimeError):
    """Base class for failures that are safe for callers to classify."""

    category: ClassVar[str] = "generation"


class DescriptionInputError(DescriptionGenerationError, ValueError):
    """Invalid caller input (class names, endpoint, or credentials)."""

    category: ClassVar[str] = "input"


class DescriptionValidationError(DescriptionGenerationError, ValueError):
    """Model output failed the local description or prompt contract."""

    category: ClassVar[str] = "validation"

    def __init__(
        self,
        message: str = "description validation failed",
        *,
        rule_code: str | None = None,
        validation_stage: str | None = None,
        row_index: int | None = None,
        field: str | None = None,
    ) -> None:
        """Create a backward-compatible error with bounded diagnostics.

        The legacy ``DescriptionValidationError(message)`` form remains valid
        and receives the ``unspecified``/``unknown`` defaults.  Structured
        values are accepted only from closed, source-defined sets; invalid
        optional values are omitted rather than copied into receipts or other
        artifacts.  In particular, these fields never contain class labels,
        generated text, response bodies, endpoints, or credentials.
        """
        self.rule_code = (
            rule_code if rule_code in DESCRIPTION_VALIDATION_RULE_CODES else "unspecified"
        )
        self.validation_stage = (
            validation_stage if validation_stage in DESCRIPTION_VALIDATION_STAGES else "unknown"
        )
        self.row_index = (
            row_index
            if isinstance(row_index, int)
            and (not isinstance(row_index, bool))
            and (0 <= row_index <= MAX_VALIDATION_ROW_INDEX)
            else None
        )
        self.field = field if field in DESCRIPTION_VALIDATION_FIELDS else None
        super().__init__(message)

    @property
    def metadata(self) -> dict[str, str | int]:
        """Return only the bounded structured fields for diagnostics."""
        values: dict[str, str | int] = {}
        if self.rule_code is not None:
            values["rule_code"] = self.rule_code
        if self.validation_stage is not None:
            values["validation_stage"] = self.validation_stage
        if self.row_index is not None:
            values["row_index"] = self.row_index
        if self.field is not None:
            values["field"] = self.field
        return values


class DescriptionDecodeError(DescriptionGenerationError, ValueError):
    """The model response could not be decoded as the required JSON object."""

    category: ClassVar[str] = "decode"


class DescriptionSchemaError(DescriptionGenerationError, ValueError):
    """The decoded JSON object did not have the required schema."""

    category: ClassVar[str] = "schema"


def _as_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(class_names, (str, bytes)):
        raise DescriptionInputError("class_names must be a sequence of labels, not a string")
    try:
        names = tuple(class_names)
    except TypeError as error:
        raise DescriptionInputError("class_names must be a finite sequence") from error
    if not names:
        raise DescriptionInputError("class_names must not be empty")
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise DescriptionInputError(f"class_names[{index}] must be a non-empty string")
        if _CONTROL_RE.search(name):
            raise DescriptionInputError(f"class_names[{index}] contains a control character")
    return names


def description_response_schema() -> dict[str, Any]:
    """Return a strict JSON schema for one batched generation request.

    A fresh object is returned on every call, preventing callers from mutating a
    shared schema used by subsequent requests.
    """
    fields = {
        "class_name": {"type": "string"},
        **{field: {"type": "string"} for field in DESCRIPTION_FIELDS},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["descriptions"],
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["class_name", *DESCRIPTION_FIELDS],
                    "properties": fields,
                },
            }
        },
    }


def build_description_prompt(class_names: Sequence[str], *, repair_prompt: bool = False) -> str:
    """Build the deterministic user prompt for one class-name batch.

    ``repair_prompt`` is a bounded validation-retry variant.  It preserves the
    response schema and exact class order while explicitly telling a provider
    not to normalize punctuation or suffixes in class names.
    """

    names = _as_class_names(class_names)
    labels = "\n".join(f"{index}. {name}" for index, name in enumerate(names, start=1))
    fields = ", ".join(DESCRIPTION_FIELDS)
    repair = (
        " This is a validation retry: copy every class_name character-for-character, "
        "including commas, suffixes, capitalization, and punctuation; do not shorten "
        "or normalize labels. The numbered input order is authoritative."
        if repair_prompt
        else ""
    )
    return (
        "For each class name below, write exactly three concise English descriptions. "
        "The fields must describe the object's visible appearance, typical functionality "
        "or behavior, and natural environment. Each description must contain 8 to 18 "
        "English words, contain only plain text, and be different from the other two "
        f"fields. Return one JSON object with a 'descriptions' array in the exact input "
        f"order. Each row must contain class_name and exactly these fields: {fields}."
        f"{repair}\n\n"
        f"Class names:\n{labels}"
    )


def build_description_messages(
    class_names: Sequence[str], *, repair_prompt: bool = False
) -> tuple[dict[str, str], ...]:
    """Return stable system/user chat messages for the structured request."""

    _as_class_names(class_names)
    system = (
        "You are a careful visual-semantic annotator. Follow the requested JSON schema "
        "exactly; never add markdown, commentary, extra keys, or reorder rows. "
        "Write natural English descriptions rather than label definitions."
    )
    return (
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_description_prompt(class_names, repair_prompt=repair_prompt),
        },
    )


def _description_words(
    value: str,
    *,
    row_index: int | None = None,
    field: str | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        raise DescriptionValidationError(
            "description fields must be non-empty strings",
            rule_code="empty",
            validation_stage="response_rows",
            row_index=row_index,
            field=field,
        )
    if _CONTROL_RE.search(value):
        raise DescriptionValidationError(
            "description contains a control character",
            rule_code="control",
            validation_stage="response_rows",
            row_index=row_index,
            field=field,
        )
    # ``split`` deliberately treats punctuation as part of a token first; the
    # regex below then rejects numerals, symbols, and non-English scripts.
    words = tuple(value.split())
    if not MIN_DESCRIPTION_WORDS <= len(words) <= MAX_DESCRIPTION_WORDS:
        raise DescriptionValidationError(
            "description word count must be between "
            f"{MIN_DESCRIPTION_WORDS} and {MAX_DESCRIPTION_WORDS}",
            rule_code="word_count",
            validation_stage="response_rows",
            row_index=row_index,
            field=field,
        )
    for word in words:
        if not _ENGLISH_WORD_RE.fullmatch(word.strip('.,;:!?()[]{}"')):
            raise DescriptionValidationError(
                "description fields must contain only English word tokens",
                rule_code="non_english",
                validation_stage="response_rows",
                row_index=row_index,
                field=field,
            )
    return words


def validate_description_rows(rows: Any, class_names: Sequence[str]) -> tuple[dict[str, str], ...]:
    """Validate and canonicalise model rows against the exact class order.

    Validation is intentionally strict and side-effect free.  A fresh tuple of
    plain dictionaries is returned, so callers can safely serialise it without
    retaining references to the model's parsed object.
    """
    expected = _as_class_names(class_names)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise DescriptionSchemaError("descriptions must be an array")
    if len(rows) != len(expected):
        raise DescriptionValidationError(
            f"expected {len(expected)} descriptions, found {len(rows)}",
            rule_code="row_count",
            validation_stage="response_rows",
        )
    canonical: list[dict[str, str]] = []
    required = {"class_name", *DESCRIPTION_FIELDS}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DescriptionSchemaError(f"descriptions[{index}] must be an object")
        if set(row) != required:
            raise DescriptionSchemaError(
                f"descriptions[{index}] must contain exactly {sorted(required)}"
            )
        class_name = row.get("class_name")
        if not isinstance(class_name, str) or class_name != expected[index]:
            raise DescriptionValidationError(
                f"class order mismatch at row {index}: expected {expected[index]!r}",
                rule_code="class_order",
                validation_stage="response_rows",
                row_index=index,
                field="class_name",
            )
        values: dict[str, str] = {"class_name": class_name}
        descriptions: list[str] = []
        for field in DESCRIPTION_FIELDS:
            value = row.get(field)
            if not isinstance(value, str):
                raise DescriptionValidationError(
                    f"description field {field!r} must be a string",
                    rule_code="field_type",
                    validation_stage="response_rows",
                    row_index=index,
                    field=field,
                )
            _description_words(value, row_index=index, field=field)
            cleaned = value.strip()
            values[field] = cleaned
            descriptions.append(cleaned)
        if len(set(descriptions)) != len(descriptions):
            raise DescriptionValidationError(
                f"description fields must be mutually different for {class_name!r}",
                rule_code="duplicate",
                validation_stage="response_rows",
                row_index=index,
            )
        canonical.append(values)
    if tuple(item["class_name"] for item in canonical) != expected:
        raise DescriptionValidationError(
            "class order check failed", rule_code="class_order", validation_stage="response_rows"
        )
    return tuple(canonical)


def _extract_json_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if not isinstance(response, Mapping):
        raise DescriptionDecodeError("response content must be a JSON string or object")
    if "choices" in response:
        choices = response.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or (not choices):
            raise DescriptionDecodeError("response choices array is empty")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise DescriptionDecodeError("response choice must be an object")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise DescriptionDecodeError("response message is missing")
        content = message.get("content")
        if isinstance(content, Sequence) and (not isinstance(content, (str, bytes))):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
                    raise DescriptionDecodeError("response content contains a non-text part")
                parts.append(part["text"])
            return "".join(parts)
        if not isinstance(content, str):
            raise DescriptionDecodeError("response message content is missing")
        return content
    return json.dumps(response, ensure_ascii=False)


def parse_description_response(response: Any) -> Any:
    """Parse a response envelope/content into a JSON object.

    Markdown fences are accepted as a compatibility repair, but validation still
    enforces the exact schema and row contract.  Raw response content is never
    included in raised error text.
    """
    text = _extract_json_text(response).strip()
    match = _FENCE_RE.fullmatch(text)
    if match:
        text = match.group(1).strip()
    try:
        decoded = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise DescriptionDecodeError("response content is not valid JSON") from error
    if not isinstance(decoded, Mapping):
        raise DescriptionSchemaError("response JSON must be an object")
    if set(decoded) != {"descriptions"}:
        raise DescriptionSchemaError("response JSON must contain only descriptions")
    return decoded
