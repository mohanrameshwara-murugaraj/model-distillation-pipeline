from distill_lab.data import dedup_train, example_id, normalize


def test_dedup_does_not_change_heldout():
    reserved = [{"text": "Why is my card payment still pending?"}]
    train = [
        {"text": "Why is my card payment still pending?"},
        {"text": "Why is my card payment still pending??"},
        {"text": "How do I freeze my card?"},
    ]
    filtered, stats = dedup_train(train, reserved)
    assert [r["text"] for r in filtered] == ["How do I freeze my card?"]
    assert len(reserved) == 1
    assert stats["removed_exact"] == 2


def test_stable_id():
    assert example_id("Hello, WORLD!") == example_id("hello world")
    assert normalize("One\nTwo") == "one two"
