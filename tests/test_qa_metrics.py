import pytest

from src.qa_metrics import exact_match_score, f1_score, normalize_answer, score_answer


@pytest.mark.parametrize("raw,expected", [
    (" The U.S.A., an Apple! A cat. ", "usa apple cat"),
    ("New-York", "newyork"),
    ("a an THE", ""),
    ("  Ada\tLovelace\n", "ada lovelace"),
    ("O'Connor", "oconnor"),
    ("“Paris”", "“paris”"),
    ("United States of America", "united states of america"),
])
def test_squad_normalization(raw, expected):
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize("prediction,gold,em,f1", [
    ("The Eiffel Tower.", "eiffel tower", 1.0, 1.0),
    ("red blue", "red green", 0.0, 0.5),
    ("red", "red blue", 0.0, 2 / 3),
    ("red blue green", "red blue", 0.0, 0.8),
    ("red red blue", "red blue blue", 0.0, 2 / 3),
    ("red red red", "red", 0.0, 0.5),
    ("Paris", "London", 0.0, 0.0),
    ("", "Paris", 0.0, 0.0),
    ("Paris", "", 0.0, 0.0),
    ("", "", 1.0, 1.0),
    ("a", "the", 1.0, 1.0),
    ("yes", "no", 0.0, 0.0),
])
def test_hand_calculated_em_and_f1(prediction, gold, em, f1):
    assert exact_match_score(prediction, gold) == em
    assert f1_score(prediction, gold) == pytest.approx(f1)


def test_alias_maximum_includes_every_alias_and_main_answer():
    assert score_answer("NYC", "New York City", ["New York", "The Big Apple", "NYC"]) == {"em": 1.0, "f1": 1.0}
    assert score_answer("New York City", "New York City", ["NYC"]) == {"em": 1.0, "f1": 1.0}
    assert score_answer("York", "Manhattan", ["New York", "New York City"]) == {"em": 0.0, "f1": pytest.approx(2 / 3)}


def test_empty_prediction_and_aliases_follow_official_musique_empty_case():
    assert score_answer("", "Paris", ["City of Paris"]) == {"em": 0.0, "f1": 0.0}
    assert score_answer("", "", ["the"]) == {"em": 1.0, "f1": 1.0}
    assert score_answer("the", "Paris", ["a"]) == {"em": 1.0, "f1": 1.0}
    assert score_answer("Paris", "Paris") == {"em": 1.0, "f1": 1.0}
    assert score_answer("Paris", "Paris", None) == {"em": 1.0, "f1": 1.0}


def test_alias_sequence_must_not_be_accidentally_scored_character_by_character():
    with pytest.raises(TypeError, match="answer_aliases"):
        score_answer("P", "London", "Paris")
