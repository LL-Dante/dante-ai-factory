"""Provider-independent tool outcomes and a deliberately bounded schema dialect."""
from enum import StrEnum
from typing import Any

from dante.contracts import StrictModel


class ToolStatus(StrEnum):
    SUCCESS = 'success'
    VALIDATION_DENIED = 'validation_denied'
    POLICY_DENIED = 'policy_denied'
    APPROVAL_REQUIRED = 'approval_required'
    APPROVAL_INVALID = 'approval_invalid'
    TIMEOUT = 'timeout'
    EXECUTION_FAILURE = 'execution_failure'
    OUTPUT_LIMIT = 'output_limit_violation'
    UNCERTAIN = 'reconciliation_required'


class ToolResult(StrictModel):
    status: ToolStatus
    data: dict[str, Any] | None = None
    effect_uncertain: bool = False
    error_type: str | None = None
    step_id: str | None = None


def check_schema(schema: dict) -> None:
    if not isinstance(schema, dict):
        raise ValueError('Schema must be an object')
    kind = schema.get('type')
    allowed = {'type', 'enum'}
    if kind == 'object':
        allowed |= {'properties', 'required', 'additionalProperties'}
        properties, required = schema.get('properties', {}), schema.get('required', [])
        if (not isinstance(properties, dict) or not isinstance(required, list)
                or not all(isinstance(key, str) for key in required)
                or not all(isinstance(key, str) for key in properties)
                or set(required) - properties.keys()
                or type(schema.get('additionalProperties', False)) is not bool):
            raise ValueError('Invalid object schema')
        for child in properties.values():
            check_schema(child)
    elif kind == 'array':
        allowed.add('items')
        check_schema(schema.get('items'))
    elif kind not in {'string', 'integer', 'number', 'boolean', 'null'}:
        raise ValueError('Unsupported schema type')
    if set(schema) - allowed or ('enum' in schema and not isinstance(schema['enum'], list)):
        raise ValueError('Unsupported schema keyword')


def validate_schema(schema: dict, value: Any) -> None:
    """Strict JSON subset; unsupported keywords fail closed, never get ignored."""
    if not isinstance(schema, dict) or set(schema) - {'type', 'properties', 'required', 'additionalProperties', 'items', 'enum'}:
        raise ValueError('Unsupported schema')
    kind = schema.get('type')
    types = {'object': dict, 'array': list, 'string': str, 'integer': int,
             'number': (int, float), 'boolean': bool, 'null': type(None)}
    if kind not in types or not isinstance(value, types[kind]):
        raise ValueError('Invalid argument type')
    if kind in {'number', 'integer'} and isinstance(value, bool):
        raise ValueError('Boolean is not a number')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('Invalid argument value')
    if kind == 'object':
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        additional = schema.get('additionalProperties', False)
        if not isinstance(properties, dict) or not isinstance(required, list) or type(additional) is not bool:
            raise ValueError('Invalid object schema')
        if not all(isinstance(key, str) for key in value) or set(required) - value.keys():
            raise ValueError('Missing argument')
        if not additional and value.keys() - properties.keys():
            raise ValueError('Unknown argument')
        for key, item in value.items():
            if key in properties:
                validate_schema(properties[key], item)
    if kind == 'array':
        if 'items' not in schema:
            raise ValueError('Missing item schema')
        for item in value:
            validate_schema(schema['items'], item)
