def test_import():
    import tricolour

    assert hasattr(tricolour, "__version__")


def test_version_is_string():
    from tricolour import __version__

    assert isinstance(__version__, str)
