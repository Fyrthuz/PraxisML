"""Script template for Alembic migration files."""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '249def1615f1'
down_revision: str | None = 'a1b2c3d4e5f6'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
