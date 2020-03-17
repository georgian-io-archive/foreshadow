from foreshadow.intents import IntentType


def test_intent_type_is_valid():
    valid_intents = [
        IntentType.CATEGORICAL,
        IntentType.NUMERIC,
        IntentType.TEXT,
        IntentType.DROPPABLE,
    ]
    for intent in valid_intents:
        assert IntentType.is_valid(intent)


def test_intent_type_not_valid():
    invalid_intent = "NOT_VALID_INTENT"
    assert not IntentType.is_valid(invalid_intent)


def test_intent_type_list_intents():
    registered_intents = IntentType.list_intents()
    valid_intents = [
        IntentType.CATEGORICAL,
        IntentType.NUMERIC,
        IntentType.TEXT,
        IntentType.DROPPABLE,
    ]

    for intent in valid_intents:
        assert intent in registered_intents
