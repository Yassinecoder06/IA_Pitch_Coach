from backend.analysis.filler_detection import count_filler_words, get_total_filler_count


def test_filler_word_detection():
    text = "Um, so basically, our product is like super fast, you know?"
    counts = count_filler_words(text)
    assert counts.get("um", 0) == 1
    assert counts.get("so", 0) == 1
    assert counts.get("basically", 0) == 1
    assert counts.get("like", 0) == 1
    assert counts.get("you know", 0) == 1
    total = get_total_filler_count(text)
    assert total >= 5


def test_clean_speech_no_fillers():
    text = "We built an autonomous database optimization platform for enterprise Postgres."
    total = get_total_filler_count(text)
    assert total == 0
