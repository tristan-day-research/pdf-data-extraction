from extraction.classifiers.pdf import PDFTypeDetector


def test_detect_digital_pdf():
    detector = PDFTypeDetector()
    assert detector.detect('data/digital.pdf') == 'digital'


def test_detect_scanned_pdf():
    detector = PDFTypeDetector()
    assert detector.detect('data/scanned.pdf') == 'scanned'
