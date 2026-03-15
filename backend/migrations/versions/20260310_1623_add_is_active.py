"""Script template for Alembic migration files."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision: str = '7f9a871a5688'
down_revision: str | None = 'b6ea64ce31c2'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # Get a connection and inspect the current state of the database
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    
    # 1. Handle 'dataset' table changes
    existing_dataset_cols = [c['name'] for c in inspector.get_columns('dataset')]
    
    # Only add 'is_active' if it's missing to avoid DuplicateColumn errors
    if 'is_active' not in existing_dataset_cols:
        op.add_column('dataset', sa.Column('is_active', sa.Boolean(), nullable=True))
    
    op.alter_column('dataset', 'is_dvc_tracked',
               existing_type=sa.BOOLEAN(),
               server_default=None,
               existing_nullable=True)
    
    # Only drop 'dvc_commit_hash' if it actually exists
    if 'dvc_commit_hash' in existing_dataset_cols:
        op.drop_column('dataset', 'dvc_commit_hash')

    # 2. Handle 'ml_model' table changes
    op.alter_column('ml_model', 'mlflow_version',
               existing_type=sa.INTEGER(),
               type_=sa.String(),
               existing_nullable=True)
    
    op.alter_column('ml_model', 'version',
               existing_type=sa.VARCHAR(),
               server_default=None,
               existing_nullable=False)
    
    op.alter_column('ml_model', 'stage',
               existing_type=sa.VARCHAR(),
               server_default=None,
               existing_nullable=False)


def downgrade() -> None:
    # 1. Revert 'ml_model' changes
    op.alter_column('ml_model', 'stage',
               existing_type=sa.VARCHAR(),
               server_default=sa.text("'Staging'::character varying"),
               existing_nullable=False)
    
    op.alter_column('ml_model', 'version',
               existing_type=sa.VARCHAR(),
               server_default=sa.text("'1.0.0'::character varying"),
               existing_nullable=False)
    
    op.alter_column('ml_model', 'mlflow_version',
               existing_type=sa.String(),
               type_=sa.INTEGER(),
               existing_nullable=True)
    
    # 2. Revert 'dataset' changes
    op.add_column('dataset', sa.Column('dvc_commit_hash', sa.VARCHAR(), autoincrement=False, nullable=True))
    
    op.alter_column('dataset', 'is_dvc_tracked',
               existing_type=sa.BOOLEAN(),
               server_default=sa.text('false'),
               existing_nullable=True)
    
    op.drop_column('dataset', 'is_active')