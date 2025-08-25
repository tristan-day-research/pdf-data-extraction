from digital_element_classifier import DigitalElementClassifier


def test_classify_detects_text():
    classifier = DigitalElementClassifier()
    result = classifier.classify('data/digital.pdf')
    assert isinstance(result, list)
    assert len(result) == 1
    page = result[0]
    assert 'text' in page and 'tables' in page and 'images' in page
    assert page['text']  # should contain words
    assert page['tables'] == []
    assert page['images'] == []
