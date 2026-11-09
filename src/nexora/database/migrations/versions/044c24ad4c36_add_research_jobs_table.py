"""Add research jobs table for durable RAG query background jobs.

Revision ID: 044c24ad4c36
Revises: 044c24ad4c35
Create Date: 2026-11-09 17:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '044c24ad4c36'
down_revision: Union[str, Sequence[str], None] = '044c24ad4c35'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add research_jobs table and indexes."""
    op.create_table(
        'research_jobs',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.Column('guest_session_id', sa.String(length=255), nullable=True),
        sa.Column('collection_id', sa.Uuid(), nullable=True),
        sa.Column('paper_id', sa.Uuid(), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('retrieval_mode', sa.String(length=32), server_default=sa.text("'hybrid'"), nullable=False),
        sa.Column('top_k', sa.Integer(), server_default=sa.text('5'), nullable=False),
        sa.Column('min_score', sa.Float(), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('max_output_tokens', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=32), server_default=sa.text("'queued'"), nullable=False),
        sa.Column('result', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('error_message', sa.String(length=2000), nullable=True),
        sa.Column('attempts', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('max_attempts', sa.Integer(), server_default=sa.text('3'), nullable=False),
        sa.Column('priority', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('claimed_by', sa.String(length=255), nullable=True),
        sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_research_jobs_user_id_users'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], name=op.f('fk_research_jobs_collection_id_collections'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], name=op.f('fk_research_jobs_paper_id_papers'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_research_jobs')),
        sa.CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed')", name=op.f('ck_research_jobs_valid_research_job_status')),
        sa.CheckConstraint('attempts >= 0', name=op.f('ck_research_jobs_research_job_attempts_non_negative')),
        sa.CheckConstraint('max_attempts > 0', name=op.f('ck_research_jobs_research_job_max_attempts_positive')),
    )
    op.create_index(op.f('ix_research_jobs_user_id'), 'research_jobs', ['user_id'], unique=False)
    op.create_index(op.f('ix_research_jobs_guest_session_id'), 'research_jobs', ['guest_session_id'], unique=False)
    op.create_index(op.f('ix_research_jobs_collection_id'), 'research_jobs', ['collection_id'], unique=False)
    op.create_index(op.f('ix_research_jobs_paper_id'), 'research_jobs', ['paper_id'], unique=False)
    op.create_index(op.f('ix_research_jobs_status'), 'research_jobs', ['status'], unique=False)
    op.create_index(op.f('ix_research_jobs_lease_expires_at'), 'research_jobs', ['lease_expires_at'], unique=False)
    op.create_index(op.f('ix_research_jobs_next_retry_at'), 'research_jobs', ['next_retry_at'], unique=False)
    op.create_index(
        'ix_research_jobs_claim',
        'research_jobs',
        ['status', 'next_retry_at', 'lease_expires_at'],
        unique=False,
    )
    op.create_index(
        'ix_research_jobs_sweeper',
        'research_jobs',
        ['status', 'lease_expires_at', 'attempts'],
        unique=False,
    )
    op.create_index(
        'ix_research_jobs_user_status',
        'research_jobs',
        ['user_id', 'status'],
        unique=False,
    )
    op.create_index(
        'ix_research_jobs_guest_status',
        'research_jobs',
        ['guest_session_id', 'status'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema: drop research_jobs table."""
    op.drop_index('ix_research_jobs_guest_status', table_name='research_jobs')
    op.drop_index('ix_research_jobs_user_status', table_name='research_jobs')
    op.drop_index('ix_research_jobs_sweeper', table_name='research_jobs')
    op.drop_index('ix_research_jobs_claim', table_name='research_jobs')
    op.drop_index(op.f('ix_research_jobs_next_retry_at'), table_name='research_jobs')
    op.drop_index(op.f('ix_research_jobs_lease_expires_at'), table_name='research_jobs')
    op.drop_index(op.f('ix_research_jobs_status'), table_name='research_jobs')
    op.drop_index(op.f('ix_research_jobs_paper_id'), table_name='research_jobs')
    op.drop_index(op.f('ix_research_jobs_collection_id'), table_name='research_jobs')
    op.drop_index(op.f('ix_research_jobs_guest_session_id'), table_name='research_jobs')
    op.drop_index(op.f('ix_research_jobs_user_id'), table_name='research_jobs')
    op.drop_table('research_jobs')
