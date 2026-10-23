import pytest

from backend.auth.owner import AuthContext, owner_service
from tests.conftest import run


def test_first_run_setup_code():
    code = owner_service.get_or_create_setup_code()
    assert len(code) == 9  # XXXX-XXXX
    assert "-" in code
    assert not run(owner_service.is_configured())


def test_owner_setup_and_login():
    code = owner_service.get_or_create_setup_code()
    with pytest.raises(ValueError):
        run(owner_service.setup("Passphrase1234", "WRONG-CODE"))

    assert run(owner_service.setup("Passphrase1234", code))
    assert run(owner_service.is_configured())

    with pytest.raises(ValueError):
        run(owner_service.setup("AnotherPassphrase", code))

    assert run(owner_service.verify_passphrase("Passphrase1234"))
    assert not run(owner_service.verify_passphrase("WrongPassphrase"))


def test_session_token_minting_and_verification(default_profile_id):
    run(owner_service.setup("ValidPassword123", owner_service.get_or_create_setup_code()))

    assert run(owner_service.verify_session_token(run(owner_service.create_session_token()))) == AuthContext(None)
    token = run(owner_service.create_session_token(default_profile_id))
    assert run(owner_service.verify_session_token(token)) == AuthContext(default_profile_id)

    assert run(owner_service.verify_session_token("invalid.token")) is None
    assert run(owner_service.verify_session_token("")) is None
    assert run(owner_service.verify_session_token(None)) is None
    payload, sig = token.split(".")
    assert run(owner_service.verify_session_token(f"{payload}.{'0' * len(sig)}")) is None


def test_change_passphrase_revokes_old_tokens():
    run(owner_service.setup("ValidPassword123", owner_service.get_or_create_setup_code()))
    old = run(owner_service.create_session_token())
    assert not run(owner_service.change_passphrase("wrong", "NewPassword456"))
    assert run(owner_service.change_passphrase("ValidPassword123", "NewPassword456"))
    assert run(owner_service.verify_session_token(old)) is None
    assert run(owner_service.verify_passphrase("NewPassword456"))
