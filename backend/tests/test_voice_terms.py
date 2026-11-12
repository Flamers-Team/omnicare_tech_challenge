from services.audio_azure_service import find_terms


def test_severe_term_is_found_ignoring_accents_and_case():
    assert find_terms("Estou com FALTA DE AR e dor no peito")["graves"] == ["falta de ar", "dor no peito"]


def test_negated_term_is_ignored():
    for phrase in ("Não tenho falta de ar", "Sem dor no peito", "Ele nega falta de ar", "Nunca tive tontura"):
        assert find_terms(phrase) == {"graves": [], "leves": []}, phrase


def test_negation_does_not_cross_sentence_or_comma():
    assert find_terms("Não tenho febre, mas estou com falta de ar")["graves"] == ["falta de ar"]
    assert find_terms("Não dormi bem. Tenho tontura")["graves"] == ["tontura"]


def test_one_non_negated_mention_is_enough():
    assert find_terms("Não senti tontura ontem, hoje estou com tontura")["graves"] == ["tontura"]


def test_mild_complaints_are_listed_but_not_severe():
    result = find_terms("Dr sinto dor de barriga e dor de garganta muito forte. E também dor de cabeça.")
    assert result["graves"] == []
    assert result["leves"] == ["dor de cabeça", "dor de barriga", "dor de garganta"]


def test_very_strong_pain_is_severe_and_listed_once():
    assert find_terms("Estou com dor muito forte")["graves"] == ["dor muito forte"]


def test_whole_word_only():
    assert find_terms("Estou descansado")["leves"] == []
