"""Add document indexing status tracking columns (Milestone 12C-22).

Revision ID: 044c24ad4c31
Revises: 044c24ad4c30
Create Date: 2026-10-13 23:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '044c24ad4c31'
down_revision: Union[str, Sequence[str], None] = '044c24ad4c30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add indexing status tracking columns to documents table."""
    op.add_column(
        'documents',
        sa.Column(
            'indexing_status',
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'completed'"),
        ),
    )
    op.add_column(
        'documents',
        sa.Column('indexing_error', sa.String(length=2000), nullable=True),
    )
    op.add_column(
        'documents',
        sa.Column('indexed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        'documents',
        sa.Column(
            'indexing_attempts',
            sa.Integer(),
            nullable=False,
            server_default=sa.text('0'),
        ),
    )
    op.create_index(
        op.f('ix_documents_indexing_status'),
        'documents',
        ['indexing_status'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema: remove indexing status tracking columns from documents table."""
    op.drop_index(op.f('ix_documents_indexing_status'), table_name='documents')
    op.drop_column('documents', 'indexing_attempts')
    op.drop_column('documents', 'indexed_at')
    op.drop_column('documents', 'indexing_error')
    op.drop_column('documents', 'indexing_status')
