from unittest.mock import patch

import pytest


class TestGetDb:
    def test_get_db_yields_session(self):
        from app.database import get_db

        gen = get_db()
        db = next(gen)
        assert db is not None
        try:
            gen.close()
        except Exception:
            pass

    def test_get_db_closes_session_on_exit(self):
        from app.database import get_db

        gen = get_db()
        db = next(gen)
        with patch.object(db, "close") as mock_close:
            try:
                next(gen)
            except StopIteration:
                pass
            mock_close.assert_called_once()

    def test_get_db_closes_on_exception(self):
        from app.database import get_db

        gen = get_db()
        db = next(gen)
        with patch.object(db, "close") as mock_close:
            try:
                gen.throw(RuntimeError, RuntimeError("test error"))
            except RuntimeError:
                pass
            mock_close.assert_called_once()

    def test_get_db_generator_closes(self):
        from app.database import get_db

        gen = get_db()
        db = next(gen)
        assert db is not None
        assert hasattr(db, "execute")
        assert hasattr(db, "close")
        try:
            gen.close()
        except Exception:
            pass


class TestSessionLocal:
    def test_session_local_is_callable(self):
        from app.database import SessionLocal

        assert callable(SessionLocal)

    def test_session_local_creates_session(self):
        from app.database import SessionLocal

        session = SessionLocal()
        assert session is not None
        assert hasattr(session, "execute")
        session.close()
