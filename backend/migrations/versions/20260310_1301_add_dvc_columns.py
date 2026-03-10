"""Script template for Alembic migration files."""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '26074141fd19'
down_revision: str | None = '022ecc4116db'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column('dataset', sa.Column('dvc_remote', sa.String(), nullable=True))
    op.add_column('dataset', sa.Column('dvc_hash', sa.String(), nullable=True))
    op.add_column('dataset', sa.Column('is_dvc_tracked', sa.Boolean(), nullable=True, server_default='False'))
    op.add_column('dataset', sa.Column('dvc_registry_name', sa.String(), nullable=True))
    op.add_column('dataset', sa.Column('dvc_version', sa.Integer(), nullable=True))
    op.add_column('dataset', sa.Column('parent_dataset_id', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('dataset', 'parent_dataset_id')
    op.drop_column('dataset', 'dvc_version')
    op.drop_column('dataset', 'dvc_registry_name')
    op.drop_column('dataset', 'is_dvc_tracked')
    op.drop_column('dataset', 'dvc_hash')
    op.drop_column('dataset', 'dvc_remote')
