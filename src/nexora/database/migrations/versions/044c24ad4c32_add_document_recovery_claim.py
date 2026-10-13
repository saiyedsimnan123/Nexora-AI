"""Add recovery claim columns to documents table (Milestone 12C-22 Follow-up).

Revision ID: 044c24ad4c32
Revises: 044c24ad4c31
Create Date: 2026-10-14 00:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '044c24ad4c32'
down_revision: Union[str, Sequence[str], None] = '044c24ad4c31'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add claimed_at and claimed_by columns to documents table."""
    op.add_column(
        'documents',
        sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        'documents',
        sa.Column('claimed_by', sa.String(length=255), nullable=True),
    )
    op.create_index(
        op.f('ix_documents_claimed_at'),
        'documents',
        ['claimed_at'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema: remove claimed_at and claimed_by columns."""
    op.drop_index(op.f('ix_documents_claimed_at'), table_name='documents')
    op.drop_column('documents', 'claimed_by')
    op.drop_column('documents', 'claimed_at')
