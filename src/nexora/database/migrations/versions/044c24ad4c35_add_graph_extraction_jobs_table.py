"""Add graph extraction jobs table (Stage 5 Step 4.6.2).

Revision ID: 044c24ad4c35
Revises: 044c24ad4c34
Create Date: 2026-11-06 00:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '044c24ad4c35'
down_revision: Union[str, Sequence[str], None] = '044c24ad4c34'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add graph_extraction_jobs table and indexes."""
    op.create_table(
        'graph_extraction_jobs',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('collection_id', sa.Uuid(), nullable=False),
        sa.Column('paper_id', sa.Uuid(), nullable=False),
        sa.Column('document_id', sa.Uuid(), nullable=True),
        sa.Column('status', sa.String(length=32), server_default=sa.text("'pending'"), nullable=False),
        sa.Column('metadata_extracted', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('priority', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('attempts', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('max_attempts', sa.Integer(), server_default=sa.text('3'), nullable=False),
        sa.Column('last_error', sa.String(length=2000), nullable=True),
        sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('claimed_by', sa.String(length=255), nullable=True),
        sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('job_metadata', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], name=op.f('fk_graph_extraction_jobs_collection_id_collections'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], name=op.f('fk_graph_extraction_jobs_paper_id_papers'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_graph_extraction_jobs_document_id_documents'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_graph_extraction_jobs')),
        sa.CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed')", name=op.f('ck_graph_extraction_jobs_valid_job_status')),
        sa.CheckConstraint('attempts >= 0', name=op.f('ck_graph_extraction_jobs_attempts_non_negative')),
        sa.CheckConstraint('max_attempts > 0', name=op.f('ck_graph_extraction_jobs_max_attempts_positive')),
    )
    op.create_index(op.f('ix_graph_extraction_jobs_collection_id'), 'graph_extraction_jobs', ['collection_id'], unique=False)
    op.create_index(op.f('ix_graph_extraction_jobs_paper_id'), 'graph_extraction_jobs', ['paper_id'], unique=False)
    op.create_index(op.f('ix_graph_extraction_jobs_document_id'), 'graph_extraction_jobs', ['document_id'], unique=False)
    op.create_index(op.f('ix_graph_extraction_jobs_status'), 'graph_extraction_jobs', ['status'], unique=False)
    op.create_index(op.f('ix_graph_extraction_jobs_lease_expires_at'), 'graph_extraction_jobs', ['lease_expires_at'], unique=False)
    op.create_index(op.f('ix_graph_extraction_jobs_next_retry_at'), 'graph_extraction_jobs', ['next_retry_at'], unique=False)
    op.create_index(
        'ix_graph_jobs_claim',
        'graph_extraction_jobs',
        ['status', 'next_retry_at', 'lease_expires_at'],
        unique=False,
    )
    op.create_index(
        'ix_graph_jobs_sweeper',
        'graph_extraction_jobs',
        ['status', 'lease_expires_at', 'attempts'],
        unique=False,
    )
    op.create_index(
        'uq_graph_jobs_active_paper',
        'graph_extraction_jobs',
        ['paper_id'],
        unique=True,
        sqlite_where=sa.text("status IN ('pending', 'processing')"),
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )


def downgrade() -> None:
    """Downgrade schema: drop graph_extraction_jobs table."""
    op.drop_index('uq_graph_jobs_active_paper', table_name='graph_extraction_jobs')
    op.drop_index('ix_graph_jobs_sweeper', table_name='graph_extraction_jobs')
    op.drop_index('ix_graph_jobs_claim', table_name='graph_extraction_jobs')
    op.drop_index(op.f('ix_graph_extraction_jobs_next_retry_at'), table_name='graph_extraction_jobs')
    op.drop_index(op.f('ix_graph_extraction_jobs_lease_expires_at'), table_name='graph_extraction_jobs')
    op.drop_index(op.f('ix_graph_extraction_jobs_status'), table_name='graph_extraction_jobs')
    op.drop_index(op.f('ix_graph_extraction_jobs_document_id'), table_name='graph_extraction_jobs')
    op.drop_index(op.f('ix_graph_extraction_jobs_paper_id'), table_name='graph_extraction_jobs')
    op.drop_index(op.f('ix_graph_extraction_jobs_collection_id'), table_name='graph_extraction_jobs')
    op.drop_table('graph_extraction_jobs')
