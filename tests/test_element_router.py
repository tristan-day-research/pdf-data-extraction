from extraction.classifiers.pdf import DigitalElementClassifier
from extraction.routers.pdf import ElementRouter


def test_route_elements_dispatches(monkeypatch):
    calls = {"text": 0, "table": 0, "image": 0}

    def fake_text(blocks):
        calls["text"] += 1

    def fake_table(table):
        calls["table"] += 1

    def fake_image(image):
        calls["image"] += 1

    monkeypatch.setattr("extraction.routers.pdf.element_router.text_processor.process_text", fake_text)
    monkeypatch.setattr("extraction.routers.pdf.element_router.table_processor.process_table", fake_table)
    monkeypatch.setattr("extraction.routers.pdf.element_router.image_processor.process_image", fake_image)

    classifier = DigitalElementClassifier()
    pages = classifier.classify('data/digital.pdf')
    router = ElementRouter()
    router.route_elements(pages)

    assert calls["text"] == 1
    assert calls["table"] == 0
    assert calls["image"] == 0
