# tests/model/test_behavioral.py
@pytest.mark.parametrize(
    "input, label",
    [
        (
            "Natural language processing is the next big wave in machine learning.",
            "natural-language-processing",
        ),
        (
            "MLOps is the next big wave in machine learning.",
            "mlops",
        ),
        (
            "This is about graph neural networks.",
            "other",
        ),
    ],
)
def test_mft(input, label, predictor):
    """Minimum Functionality Tests (simple input/output pairs)."""
    prediction = utils.get_label(text=input, predictor=predictor)
    assert label == prediction
