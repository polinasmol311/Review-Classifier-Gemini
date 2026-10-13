from src.classifier import parse_response


def test_valid_json():
    r = parse_response('{"sentiment":"negative","topic":"delivery","urgency":"high","summary":"Late"}')
    assert r["sentiment"] == "negative" and r["topic"] == "delivery"


def test_json_wrapped_in_text():
    r = parse_response('Sure! {"sentiment":"positive","topic":"price","urgency":"low"}')
    assert r["sentiment"] == "positive"


def test_garbage_falls_back():
    assert parse_response("no json here")["sentiment"] == "neutral"


def test_invalid_label_falls_back():
    assert parse_response('{"sentiment":"angry"}')["sentiment"] == "neutral"
