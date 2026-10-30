"""Declarative serializers: a projection between Python objects and JSON.

Fields are declared either explicitly (``tier = IntField(min_value=0)``) or as
annotations carrying a :class:`Field` metadata object (``tier: int = Field(min_value=0)``).
Both forms resolve to the same internal field map, which is what the OpenAPI
generator and the validation code read.
"""

from __future__ import annotations

import copy
import inspect
import math
import re
import types
import typing
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Union

from .exceptions import ValidationError

__all__ = [
    "BoolField",
    "CharField",
    "ChoiceField",
    "DictField",
    "Field",
    "FieldError",
    "FloatField",
    "IntField",
    "ListField",
    "MISSING",
    "ModelSerializer",
    "NestedField",
    "RawField",
    "Serializer",
    "ValidationError",
]


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _Missing()

_OPTION_NAMES = {
    "required",
    "default",
    "default_factory",
    "allow_null",
    "read_only",
    "help_text",
    "min_value",
    "max_value",
    "min_length",
    "max_length",
    "pattern",
    "choices",
    "case_sensitive",
    "allow_empty",
    "child",
    "value_field",
    "serializer",
    "many",
}


class FieldError(Exception):
    """A single field's failure: a list of messages, or a nested ``{key: [...]}`` dict."""

    def __init__(self, detail: str | list[str] | dict[str, Any]) -> None:
        if isinstance(detail, str):
            detail = [detail]
        self.detail = detail
        super().__init__(str(detail))


class Field:
    """Base field and the annotated-form metadata object.

    ``Field(max_length=16)`` on an annotated attribute records options; the
    annotation picks the concrete class. Concrete classes accept the same options.
    """

    json_type: str | None = None
    json_format: str | None = None

    def __init__(
        self,
        *,
        required: bool | None = None,
        default: Any = MISSING,
        default_factory: Callable[[], Any] | None = None,
        allow_null: bool = False,
        read_only: bool = False,
        help_text: str = "",
        min_value: float | None = None,
        max_value: float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        choices: Iterable[Any] | None = None,
        case_sensitive: bool = True,
        allow_empty: bool = True,
        child: "Field | None" = None,
        value_field: "Field | None" = None,
        serializer: "type[Serializer] | None" = None,
        many: bool = False,
    ) -> None:
        if default is not MISSING and default_factory is not None:
            raise TypeError("a field takes default or default_factory, not both")
        self.default = default
        self.default_factory = default_factory
        self.allow_null = bool(allow_null)
        self.read_only = bool(read_only)
        self.help_text = help_text
        self.min_value = min_value
        self.max_value = max_value
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self._pattern = re.compile(pattern) if pattern else None
        self.choices = list(choices) if choices is not None else None
        self.case_sensitive = bool(case_sensitive)
        self.allow_empty = bool(allow_empty)
        self.child = child
        self.value_field = value_field
        self.serializer = serializer
        self.many = bool(many)
        self._explicit_required = required
        self.name = ""
        self._options = {
            key: value
            for key, value in dict(
                required=required,
                default=default,
                default_factory=default_factory,
                allow_null=allow_null,
                read_only=read_only,
                help_text=help_text,
                min_value=min_value,
                max_value=max_value,
                min_length=min_length,
                max_length=max_length,
                pattern=pattern,
                choices=choices,
                case_sensitive=case_sensitive,
                allow_empty=allow_empty,
                child=child,
                value_field=value_field,
                serializer=serializer,
                many=many,
            ).items()
            if _was_set(key, value)
        }

    # -- metadata -----------------------------------------------------------
    @property
    def required(self) -> bool:
        if self.read_only:
            return False
        if self._explicit_required is not None:
            return bool(self._explicit_required)
        return not self.has_default

    @property
    def has_default(self) -> bool:
        return self.default is not MISSING or self.default_factory is not None

    def get_default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        value = self.default
        return copy.deepcopy(value) if isinstance(value, (list, dict, set)) else value

    def bind(self, name: str) -> None:
        self.name = name

    # -- input ---------------------------------------------------------------
    def run_validation(self, value: Any) -> Any:
        if value is None:
            if self.allow_null:
                return None
            raise FieldError("this field may not be null.")
        return self.to_internal(value)

    def to_internal(self, value: Any) -> Any:
        return value

    # -- output --------------------------------------------------------------
    def to_representation(self, value: Any) -> Any:
        return value

    # -- schema ----------------------------------------------------------------
    def schema(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.json_type:
            out["type"] = self.json_type
        if self.json_format:
            out["format"] = self.json_format
        out.update(self.extra_schema())
        if self.help_text:
            out["description"] = self.help_text
        if self.allow_null:
            out["nullable"] = True
        if self.read_only:
            out["readOnly"] = True
        if self.default is not MISSING and _json_safe(self.default):
            out["default"] = self.default
        return out

    def extra_schema(self) -> dict[str, Any]:
        return {}

    def describe(self) -> dict[str, Any]:
        """The four attributes OpenAPI reads: required, default, help_text, read_only."""
        return {
            "required": self.required,
            "default": None if not self.has_default else self.get_default(),
            "help_text": self.help_text,
            "read_only": self.read_only,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def _was_set(key: str, value: Any) -> bool:
    defaults: dict[str, Any] = {
        "required": None,
        "default": MISSING,
        "default_factory": None,
        "allow_null": False,
        "read_only": False,
        "help_text": "",
        "min_value": None,
        "max_value": None,
        "min_length": None,
        "max_length": None,
        "pattern": None,
        "choices": None,
        "case_sensitive": True,
        "allow_empty": True,
        "child": None,
        "value_field": None,
        "serializer": None,
        "many": False,
    }
    return value is not defaults[key] and value != defaults[key]


def _json_safe(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, list, dict, type(None)))


class CharField(Field):
    json_type = "string"

    def __init__(
        self,
        max_length: int | None = None,
        min_length: int | None = None,
        pattern: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(max_length=max_length, min_length=min_length, pattern=pattern, **options)

    def to_internal(self, value: Any) -> str:
        if not isinstance(value, str):
            raise FieldError("not a valid string.")
        if self.min_length is not None and len(value) < self.min_length:
            raise FieldError(f"ensure this value has at least {self.min_length} characters.")
        if self.max_length is not None and len(value) > self.max_length:
            raise FieldError(f"ensure this value has at most {self.max_length} characters.")
        if self._pattern is not None and self._pattern.search(value) is None:
            raise FieldError(f"this value does not match the pattern {self.pattern!r}.")
        return value

    def to_representation(self, value: Any) -> Any:
        return value if value is None else str(value)

    def extra_schema(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.min_length is not None:
            out["minLength"] = self.min_length
        if self.max_length is not None:
            out["maxLength"] = self.max_length
        if self.pattern:
            out["pattern"] = self.pattern
        return out


class _Numeric(Field):
    def _bounds(self, value: float) -> None:
        if self.min_value is not None and value < self.min_value:
            raise FieldError(f"ensure this value is greater than or equal to {self.min_value:g}.")
        if self.max_value is not None and value > self.max_value:
            raise FieldError(f"ensure this value is less than or equal to {self.max_value:g}.")

    def extra_schema(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.min_value is not None:
            out["minimum"] = self.min_value
        if self.max_value is not None:
            out["maximum"] = self.max_value
        return out


class IntField(_Numeric):
    json_type = "integer"

    def __init__(
        self, min_value: int | None = None, max_value: int | None = None, **options: Any
    ) -> None:
        super().__init__(min_value=min_value, max_value=max_value, **options)

    def to_internal(self, value: Any) -> int:
        # ``bool`` is an ``int``: accepting it would turn ``{"episode": true}`` into episode 1.
        if isinstance(value, bool):
            raise FieldError("a boolean is not a valid integer.")
        if isinstance(value, int):
            number = value
        elif isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                raise FieldError("a whole number is required.")
            number = int(value)
        elif isinstance(value, str) and re.fullmatch(r"\s*[+-]?\d+\s*", value):
            number = int(value)
        else:
            raise FieldError("a valid integer is required.")
        self._bounds(number)
        return number

    def to_representation(self, value: Any) -> Any:
        return value if value is None else int(value)


class FloatField(_Numeric):
    json_type = "number"

    def __init__(
        self, min_value: float | None = None, max_value: float | None = None, **options: Any
    ) -> None:
        super().__init__(min_value=min_value, max_value=max_value, **options)

    def to_internal(self, value: Any) -> float:
        if isinstance(value, bool):
            raise FieldError("a boolean is not a valid number.")
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, str):
            try:
                number = float(value.strip())
            except ValueError:
                raise FieldError("a valid number is required.") from None
        else:
            raise FieldError("a valid number is required.")
        if not math.isfinite(number):
            raise FieldError("a finite number is required.")
        self._bounds(number)
        return number

    def to_representation(self, value: Any) -> Any:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None


class BoolField(Field):
    json_type = "boolean"
    _TRUE = {"true", "1"}
    _FALSE = {"false", "0"}

    def to_internal(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in self._TRUE:
                return True
            if lowered in self._FALSE:
                return False
        raise FieldError("must be a valid boolean (true/false/1/0).")

    def to_representation(self, value: Any) -> Any:
        return value if value is None else bool(value)


class ChoiceField(Field):
    json_type = "string"

    def __init__(self, choices: Iterable[Any], *, case_sensitive: bool = True, **options: Any) -> None:
        super().__init__(choices=list(choices), case_sensitive=case_sensitive, **options)
        if not self.choices:
            raise TypeError("ChoiceField needs at least one choice")

    def to_internal(self, value: Any) -> Any:
        assert self.choices is not None
        if not isinstance(value, str):
            if value in self.choices:
                return value
            raise FieldError(f"{value!r} is not a valid choice; expected one of {self.choices}.")
        if value in self.choices:
            return value
        if not self.case_sensitive:
            lowered = value.lower()
            for choice in self.choices:
                if isinstance(choice, str) and choice.lower() == lowered:
                    return choice
        raise FieldError(f"{value!r} is not a valid choice; expected one of {self.choices}.")

    def extra_schema(self) -> dict[str, Any]:
        assert self.choices is not None
        return {"enum": list(self.choices)}


class ListField(Field):
    json_type = "array"

    def __init__(self, child: Field | None = None, *, allow_empty: bool = True, **options: Any) -> None:
        super().__init__(child=child or RawField(), allow_empty=allow_empty, **options)

    def to_internal(self, value: Any) -> list[Any]:
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
            raise FieldError("expected a list of items.")
        if not value and not self.allow_empty:
            raise FieldError("this list may not be empty.")
        if self.min_length is not None and len(value) < self.min_length:
            raise FieldError(f"ensure this list has at least {self.min_length} items.")
        if self.max_length is not None and len(value) > self.max_length:
            raise FieldError(f"ensure this list has at most {self.max_length} items.")
        assert self.child is not None
        out: list[Any] = []
        errors: dict[str, Any] = {}
        for index, item in enumerate(value):
            try:
                out.append(self.child.run_validation(item))
            except FieldError as exc:
                errors[str(index)] = exc.detail
        if errors:
            raise FieldError(errors)
        return out

    def to_representation(self, value: Any) -> Any:
        if value is None:
            return None
        assert self.child is not None
        return [self.child.to_representation(item) for item in value]

    def extra_schema(self) -> dict[str, Any]:
        assert self.child is not None
        out: dict[str, Any] = {"items": self.child.schema()}
        if not self.allow_empty:
            out["minItems"] = 1
        return out


class DictField(Field):
    json_type = "object"

    def __init__(self, value_field: Field | None = None, **options: Any) -> None:
        super().__init__(value_field=value_field, **options)

    def to_internal(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise FieldError("expected an object.")
        if not all(isinstance(key, str) for key in value):
            raise FieldError("object keys must be strings.")
        if self.value_field is None:
            return dict(value)
        out: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        for key, item in value.items():
            try:
                out[key] = self.value_field.run_validation(item)
            except FieldError as exc:
                errors[key] = exc.detail
        if errors:
            raise FieldError(errors)
        return out

    def to_representation(self, value: Any) -> Any:
        if value is None:
            return None
        if self.value_field is None:
            return dict(value)
        return {key: self.value_field.to_representation(item) for key, item in value.items()}

    def extra_schema(self) -> dict[str, Any]:
        return {"additionalProperties": self.value_field.schema() if self.value_field else True}


class NestedField(Field):
    json_type = "object"

    def __init__(self, serializer: "type[Serializer]", many: bool = False, **options: Any) -> None:
        super().__init__(serializer=serializer, many=many, **options)

    def to_internal(self, value: Any) -> Any:
        assert self.serializer is not None
        nested = self.serializer(data=value, many=self.many)
        if not nested.is_valid():
            raise FieldError(nested.errors)
        return nested.validated_data

    def to_representation(self, value: Any) -> Any:
        if value is None:
            return None
        assert self.serializer is not None
        return self.serializer(instance=value, many=self.many).to_representation()

    def schema(self) -> dict[str, Any]:
        assert self.serializer is not None
        inner = self.serializer.schema()
        out = {"type": "array", "items": inner} if self.many else dict(inner)
        if self.help_text:
            out["description"] = self.help_text
        if self.allow_null:
            out["nullable"] = True
        if self.read_only:
            out["readOnly"] = True
        return out


class RawField(Field):
    """Any JSON value, passed through unchanged."""

    def to_internal(self, value: Any) -> Any:
        return value


# ---------------------------------------------------------------------------
# Annotation resolution
# ---------------------------------------------------------------------------
_RESERVED = {"unknown_fields", "model", "many", "instance", "initial_data"}
_NONE_TYPE = type(None)


def _field_from_hint(hint: Any, options: Mapping[str, Any]) -> Field:
    options = dict(options)
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Annotated:
        return _field_from_hint(args[0], options)
    if origin in (Union, getattr(types, "UnionType", Union)):
        non_none = [arg for arg in args if arg is not _NONE_TYPE]
        optional = len(non_none) != len(args)
        if optional:
            options.setdefault("allow_null", True)
            if "default" not in options and "default_factory" not in options:
                options["default"] = None
        if len(non_none) == 1:
            return _field_from_hint(non_none[0], options)
        return RawField(**_only(options, {"required", "default", "default_factory", "allow_null",
                                         "read_only", "help_text"}))
    if origin is Literal:
        return ChoiceField(list(args), **_only(options, _COMMON | {"case_sensitive"}))
    if origin in (list, tuple, set, frozenset, Sequence, typing.Sequence, typing.List, typing.Tuple):
        element = args[0] if args else Any
        if inspect.isclass(element) and issubclass(element, Serializer):
            return NestedField(element, many=True, **_only(options, _COMMON))
        return ListField(
            _field_from_hint(element, {}),
            **_only(options, _COMMON | {"allow_empty", "min_length", "max_length"}),
        )
    if origin in (dict, Mapping, typing.Mapping, typing.Dict):
        value = _field_from_hint(args[1], {}) if len(args) == 2 and args[1] is not Any else None
        return DictField(value, **_only(options, _COMMON))
    if options.get("choices") is not None and hint in (str, int, float, bool):
        # ``kind: str = Field(choices=[...])``: the enum is the constraint that matters.
        return ChoiceField(options["choices"], **_only(options, _COMMON | {"case_sensitive"}))
    if hint is str:
        return CharField(**_only(options, _COMMON | {"min_length", "max_length", "pattern"}))
    if hint is bool:
        return BoolField(**_only(options, _COMMON))
    if hint is int:
        return IntField(**_only(options, _COMMON | {"min_value", "max_value"}))
    if hint is float:
        return FloatField(**_only(options, _COMMON | {"min_value", "max_value"}))
    if hint is dict:
        return DictField(**_only(options, _COMMON))
    if hint is list:
        return ListField(**_only(options, _COMMON | {"allow_empty"}))
    if inspect.isclass(hint) and issubclass(hint, Serializer):
        return NestedField(hint, **_only(options, _COMMON))
    if options.get("choices"):
        return ChoiceField(options["choices"], **_only(options, _COMMON | {"case_sensitive"}))
    return RawField(**_only(options, _COMMON))


_COMMON = {"required", "default", "default_factory", "allow_null", "read_only", "help_text"}


def _only(options: Mapping[str, Any], names: set[str]) -> dict[str, Any]:
    return {key: value for key, value in options.items() if key in names}


def _own_annotations(cls: type) -> dict[str, Any]:
    """Evaluated annotations declared on ``cls`` itself (not its bases)."""
    try:
        raw = inspect.get_annotations(cls)
    except Exception:  # noqa: BLE001
        raw = dict(cls.__dict__.get("__annotations__", {}))
    if not raw:
        return {}
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:  # noqa: BLE001 - an unresolved forward ref: fall back per name
        hints = {}
    resolved: dict[str, Any] = {}
    module = __import__("sys").modules.get(cls.__module__)
    namespace = dict(vars(module)) if module else {}
    for name, annotation in raw.items():
        if name in hints:
            resolved[name] = hints[name]
        elif isinstance(annotation, str):
            try:
                resolved[name] = eval(annotation, namespace, dict(vars(cls)))  # noqa: S307
            except Exception as exc:  # noqa: BLE001
                raise TypeError(
                    f"{cls.__name__}.{name}: cannot resolve annotation {annotation!r}: {exc}"
                ) from exc
        else:
            resolved[name] = annotation
    return resolved


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------
_UNSET: Any = object()


class Serializer:
    """Validate input, project output, publish a schema."""

    #: ``"forbid"`` rejects unknown input keys (request bodies); ``"allow"`` copies
    #: them through (response bodies from a newer library).
    unknown_fields: Literal["forbid", "allow"] = "allow"

    def __init__(self, instance: Any = None, *, data: Any = _UNSET, many: bool = False) -> None:
        self.instance = instance
        self.initial_data = data
        self.many = bool(many)
        self._errors: Any = None
        self._validated: Any = _UNSET
        for field in self.__class__._get_fields().values():
            # A field whose choices come from a live registry re-reads them per instance
            # (document 17 section 16.3.2); every other field class has no ``refresh``.
            refresh = getattr(field, "refresh", None)
            if callable(refresh):
                refresh()

    # -- declaration ------------------------------------------------------------
    @classmethod
    def _get_fields(cls) -> dict[str, Field]:
        cached = cls.__dict__.get("_resolved_fields")
        if cached is not None:
            return cached
        fields: dict[str, Field] = {}
        for base in reversed(cls.__mro__[1:]):
            if base is not Serializer and isinstance(base, type) and issubclass(base, Serializer):
                for name, field in base._get_fields().items():
                    fields[name] = copy.copy(field)
        annotations = _own_annotations(cls)
        namespace = cls.__dict__
        ordered = list(annotations) + [
            name for name, value in namespace.items() if isinstance(value, Field) and name not in annotations
        ]
        for name in ordered:
            if name.startswith("_"):
                continue
            # A reserved name is a declaration only when explicitly given a ``Field``
            # (``model: str = Field()`` on a spec); the plain class attribute is not.
            if name in _RESERVED and not isinstance(namespace.get(name), Field):
                continue
            value = namespace.get(name, MISSING)
            hint = annotations.get(name, MISSING)
            if hint is not MISSING and typing.get_origin(hint) is typing.ClassVar:
                continue
            if isinstance(value, Field) and type(value) is not Field:
                field = copy.copy(value)
            else:
                if hint is MISSING:
                    continue
                if isinstance(value, Field):
                    options = dict(value._options)
                elif value is MISSING:
                    options = {}
                elif callable(value) and not isinstance(value, (str, int, float, bool)):
                    continue  # a method that happens to share an annotation name
                else:
                    options = {"default": value}
                field = _field_from_hint(hint, options)
            field.bind(name)
            fields[name] = field
        type.__setattr__(cls, "_resolved_fields", fields)
        return fields

    @property
    def fields(self) -> dict[str, Field]:
        return self._get_fields()

    # -- validation ---------------------------------------------------------------
    def is_valid(self, *, raise_exception: bool = False) -> bool:
        if self.initial_data is _UNSET:
            raise TypeError("is_valid() needs the serializer to be built with data=...")
        if self._validated is _UNSET:
            self._run()
        if self._errors and raise_exception:
            raise ValidationError(self._errors)
        return not self._errors

    def _run(self) -> None:
        if self.many:
            self._validate_many()
            return
        try:
            self._validated = self._validate_one(self.initial_data)
            self._errors = {}
        except FieldError as exc:
            self._validated = {}
            self._errors = exc.detail if isinstance(exc.detail, dict) else {"non_field_errors": exc.detail}

    def _validate_many(self) -> None:
        data = self.initial_data
        if isinstance(data, (str, bytes, Mapping)) or not isinstance(data, (list, tuple)):
            self._validated, self._errors = [], {"non_field_errors": ["expected a list of items."]}
            return
        out, errors = [], {}
        for index, item in enumerate(data):
            try:
                out.append(self._validate_one(item))
            except FieldError as exc:
                errors[str(index)] = exc.detail
        self._validated, self._errors = out, errors

    def _validate_one(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise FieldError({"non_field_errors": ["expected an object."]})
        fields = self._get_fields()
        errors: dict[str, Any] = {}
        attrs: dict[str, Any] = {}
        for name, field in fields.items():
            if field.read_only:
                continue
            if name in data:
                try:
                    value = field.run_validation(data[name])
                    hook = getattr(self, f"validate_{name}", None)
                    if callable(hook):
                        value = hook(value)
                    attrs[name] = value
                except FieldError as exc:
                    errors[name] = exc.detail
            elif field.has_default:
                attrs[name] = field.get_default()
            elif field.required:
                errors[name] = ["this field is required."]
        unknown = [key for key in data if key not in fields]
        if unknown:
            if self.unknown_fields == "forbid":
                errors.setdefault("non_field_errors", []).extend(
                    f"unexpected field: {key!r}" for key in unknown
                )
            else:
                for key in unknown:
                    attrs[key] = data[key]
        if errors:
            raise FieldError(errors)
        try:
            attrs = self.validate(attrs)
        except FieldError as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"non_field_errors": exc.detail}
            raise FieldError(detail) from None
        return attrs

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        """Cross-field hook. Raise :class:`FieldError` to reject."""
        return attrs

    @property
    def errors(self) -> dict[str, list[str]]:
        if self._validated is _UNSET and self.initial_data is not _UNSET:
            self._run()
        return self._errors or {}

    @property
    def validated_data(self) -> Any:
        if self._validated is _UNSET:
            raise AssertionError("call is_valid() before reading validated_data")
        if self._errors:
            raise AssertionError("validated_data is unavailable: the data is invalid")
        return self._validated

    # -- output ----------------------------------------------------------------------
    def to_representation(self) -> Any:
        if self.many:
            return [self._represent(item) for item in (self.instance or [])]
        return self._represent(self.instance)

    def _represent(self, instance: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        fields = self._get_fields()
        for name, field in fields.items():
            found, value = _get_attribute(instance, name)
            if not found:
                if field.has_default:
                    value = field.get_default()
                else:
                    continue
            out[name] = None if value is None else field.to_representation(value)
        if self.unknown_fields == "allow" and isinstance(instance, Mapping):
            for key, value in instance.items():
                if key not in fields:
                    out[key] = value
        return out

    def to_dict(self) -> Any:
        return self.to_representation()

    @property
    def data(self) -> Any:
        if self.instance is not None:
            return self.to_representation()
        return self.validated_data

    def save(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} is not a ModelSerializer")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Serializer":
        serializer = cls(data=payload)
        serializer.is_valid(raise_exception=True)
        return serializer

    @classmethod
    def build(cls, obj: Any) -> "Serializer":
        return cls(instance=obj)

    # -- schema ---------------------------------------------------------------------------
    @classmethod
    def schema(cls) -> dict[str, Any]:
        fields = cls._get_fields()
        properties = {name: field.schema() for name, field in fields.items()}
        required = [name for name, field in fields.items() if field.required]
        out: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            out["required"] = required
        out["additionalProperties"] = cls.unknown_fields != "forbid"
        if cls.__doc__:
            out["description"] = inspect.cleandoc(cls.__doc__).split("\n\n", 1)[0]
        return out


def _get_attribute(instance: Any, name: str) -> tuple[bool, Any]:
    if isinstance(instance, Mapping):
        if name in instance:
            return True, instance[name]
        return False, None
    if hasattr(instance, name):
        return True, getattr(instance, name)
    return False, None


class ModelSerializer(Serializer):
    """Binds a framework dataclass through its own ``to_dict``/``from_dict``.

    The framework class is the single source of truth: there is no parallel model
    family. A subclass exists only when the wire shape differs from ``to_dict()``,
    in which case it overrides :meth:`to_representation` and calls ``super()``.
    """

    model: type | None = None

    def _represent(self, instance: Any) -> dict[str, Any]:
        to_dict = getattr(instance, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if isinstance(instance, Mapping):
            return dict(instance)
        return super()._represent(instance)

    def save(self) -> Any:
        if self.model is None:
            raise TypeError(f"{type(self).__name__}.model is not set")
        data = self.validated_data
        factory = getattr(self.model, "from_dict", None)
        if self.many:
            return [factory(item) if callable(factory) else self.model(**item) for item in data]
        return factory(data) if callable(factory) else self.model(**data)

    @classmethod
    def build(cls, obj: Any) -> "ModelSerializer":
        return cls(instance=obj)
