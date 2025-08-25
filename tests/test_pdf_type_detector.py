from classifiers.pdf import PdfTypeDetector


def test_detect_digital_pdf():
    detector = PdfTypeDetector()
    assert detector.detect('data/digital.pdf') == 'digital'


def test_detect_scanned_pdf():
    detector = PdfTypeDetector()
    assert detector.detect('data/scanned.pdf') == 'scanned'
