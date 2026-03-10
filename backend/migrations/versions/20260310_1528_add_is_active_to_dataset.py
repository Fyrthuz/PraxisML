"""Script template for Alembic migration files."""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b6ea64ce31c2'
down_revision: str | None = '26074141fd19'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column('dataset', sa.Column('is_active', sa.Boolean(), nullable=True, server_default='False'))


def downgrade() -> None:
    op.drop_column('dataset', 'is_active')
