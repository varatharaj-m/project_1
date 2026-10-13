from app.services.file_validation import ValidationError, validate_extension


def test_validate_extension_rejects_unknown():
    try:
        validate_extension("bad.exe", {".png"})
    except ValidationError:
        assert True
    else:
        assert False
