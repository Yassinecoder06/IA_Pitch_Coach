import os
import pytest
from backend.storage.database import db
from backend.auth.owner import owner_service, DEFAULT_LOCAL_USER_ID


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    test_db = tmp_path / "test.db"
    db.sqlite_file = test_db
    db.is_postgres = False
    db.init_schema()
    owner_service._setup_code = None


def test_first_run_setup_code():
    code = owner_service.get_or_create_setup_code()
    assert len(code) == 9  # XXXX-XXXX
    assert "-" in code
    assert not owner_service.is_configured()


def test_owner_setup_and_login():
    code = owner_service.get_or_create_setup_code()
    # Wrong code fails
    with pytest.raises(ValueError):
        owner_service.setup("Passphrase1234", "WRONG-CODE")

    # Correct code succeeds
    assert owner_service.setup("Passphrase1234", code)
    assert owner_service.is_configured()

    # Second setup fails
    with pytest.raises(ValueError):
        owner_service.setup("AnotherPassphrase", code)

    # Login verification
    assert owner_service.verify_passphrase("Passphrase1234")
    assert not owner_service.verify_passphrase("WrongPassphrase")


def test_session_token_minting_and_verification():
    code = owner_service.get_or_create_setup_code()
    owner_service.setup("ValidPassword123", code)

    token = owner_service.create_session_token()
    user_id = owner_service.verify_session_token(token)
    assert user_id == DEFAULT_LOCAL_USER_ID

    # Malformed tokens fail
    assert owner_service.verify_session_token("invalid.token") is None
    assert owner_service.verify_session_token("") is None
    assert owner_service.verify_session_token(None) is None
