from datetime import datetime

from seqtrainer.sparql.normalize import normalize_bindings
from seqtrainer.sparql.recipes import (
    component_sequences_recipe,
    list_recipes,
    numerical_measurements_recipe,
    sequence_annotations_recipe,
)


def test_numerical_recipe_has_float_schema_and_query():
    recipe = numerical_measurements_recipe()
    assert recipe.schema["value"] == "float"
    assert "hasNumericalValue" in recipe.query


def test_component_sequences_recipe_contains_expected_patterns():
    recipe = component_sequences_recipe(limit=25)
    assert "sbol:ComponentDefinition" in recipe.query
    assert "sbol:elements" in recipe.query
    assert "LIMIT 25" in recipe.query


def test_sequence_annotations_recipe_schema_types():
    recipe = sequence_annotations_recipe()
    assert recipe.schema["start"] == "int"
    assert recipe.schema["end"] == "int"


def test_normalize_bindings_applies_typed_schema():
    payload = {
        "results": {
            "bindings": [
                {
                    "subject": {"type": "uri", "value": "https://example.org/s1"},
                    "value": {
                        "type": "literal",
                        "datatype": "http://www.w3.org/2001/XMLSchema#double",
                        "value": "3.14",
                    },
                    "start": {
                        "type": "literal",
                        "datatype": "http://www.w3.org/2001/XMLSchema#integer",
                        "value": "5",
                    },
                    "timestamp": {
                        "type": "literal",
                        "datatype": "http://www.w3.org/2001/XMLSchema#dateTime",
                        "value": "2026-01-01T12:30:00Z",
                    },
                }
            ]
        }
    }
    rows = normalize_bindings(
        payload,
        schema={"subject": "uri", "value": "float", "start": "int", "timestamp": "datetime"},
    )
    assert rows[0]["subject"] == "https://example.org/s1"
    assert rows[0]["value"] == 3.14
    assert rows[0]["start"] == 5
    assert isinstance(rows[0]["timestamp"], datetime)


def test_recipe_registry_contains_canonical_recipes():
    registry = list_recipes()
    assert "component-definitions" in registry
    assert "component-sequences" in registry
    assert "sequence-annotations" in registry
