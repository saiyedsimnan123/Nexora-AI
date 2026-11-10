"""Add research jobs table for durable RAG query background jobs.

Revision ID: 044c24ad4c36
Revises: 044c24ad4c35
Create Date: 2026-11-09 17:45:00.000000

"""
from __future__ import annotations

import logging
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = '044c24ad4c36'
down_revision: Union[str, Sequence[str], None] = '044c24ad4c35'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _inspect_pg_types(bind: sa.Connection) -> set[str]:
    """Return existing PostgreSQL type names in the current schema for diagnostics."""
    if bind.dialect.name != 'postgresql':
        return set()
    rows = bind.execute(
        sa.text(
            "SELECT t.typname "
            "FROM pg_type t "
            "JOIN pg_namespace n ON n.oid = t.typnamespace "
            "WHERE n.nspname = current_schema() "
            "AND t.typname IN ('research_jobs', 'researchjobstatus')"
        )
    ).scalars().all()
    return {str(name) for name in rows}


def _expected_column_definitions(dialect_name: str) -> list[tuple[str, sa.Column]]:
    """Return ordered optional/evolvable column definitions for partial-table recovery."""
    ts_col_created = (
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True)
        if dialect_name == 'sqlite'
        else sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False)
    )
    ts_col_updated = (
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True)
        if dialect_name == 'sqlite'
        else sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False)
    )
    return [
        ('user_id', sa.Column('user_id', sa.Uuid(), nullable=True)),
        ('guest_session_id', sa.Column('guest_session_id', sa.String(length=255), nullable=True)),
        ('collection_id', sa.Column('collection_id', sa.Uuid(), nullable=True)),
        ('paper_id', sa.Column('paper_id', sa.Uuid(), nullable=True)),
        ('retrieval_mode', sa.Column('retrieval_mode', sa.String(length=32), server_default=sa.text("'hybrid'"), nullable=False)),
        ('top_k', sa.Column('top_k', sa.Integer(), server_default=sa.text('5'), nullable=False)),
        ('min_score', sa.Column('min_score', sa.Float(), nullable=True)),
        ('temperature', sa.Column('temperature', sa.Float(), nullable=True)),
        ('max_output_tokens', sa.Column('max_output_tokens', sa.Integer(), nullable=True)),
        ('status', sa.Column('status', sa.String(length=32), server_default=sa.text("'queued'"), nullable=False)),
        ('result', sa.Column('result', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True)),
        ('error_message', sa.Column('error_message', sa.String(length=2000), nullable=True)),
        ('attempts', sa.Column('attempts', sa.Integer(), server_default=sa.text('0'), nullable=False)),
        ('max_attempts', sa.Column('max_attempts', sa.Integer(), server_default=sa.text('3'), nullable=False)),
        ('priority', sa.Column('priority', sa.Integer(), server_default=sa.text('0'), nullable=False)),
        ('claimed_by', sa.Column('claimed_by', sa.String(length=255), nullable=True)),
        ('claimed_at', sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True)),
        ('lease_expires_at', sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True)),
        ('next_retry_at', sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True)),
        ('completed_at', sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True)),
        ('created_at', ts_col_created),
        ('updated_at', ts_col_updated),
    ]


def _expected_index_definitions() -> list[tuple[str, list[str]]]:
    """Return index definitions for research_jobs."""
    return [
        (op.f('ix_research_jobs_user_id'), ['user_id']),
        (op.f('ix_research_jobs_guest_session_id'), ['guest_session_id']),
        (op.f('ix_research_jobs_collection_id'), ['collection_id']),
        (op.f('ix_research_jobs_paper_id'), ['paper_id']),
        (op.f('ix_research_jobs_status'), ['status']),
        (op.f('ix_research_jobs_lease_expires_at'), ['lease_expires_at']),
        (op.f('ix_research_jobs_next_retry_at'), ['next_retry_at']),
        ('ix_research_jobs_claim', ['status', 'next_retry_at', 'lease_expires_at']),
        ('ix_research_jobs_sweeper', ['status', 'lease_expires_at', 'attempts']),
        ('ix_research_jobs_user_status', ['user_id', 'status']),
        ('ix_research_jobs_guest_status', ['guest_session_id', 'status']),
    ]


def upgrade() -> None:
    """Upgrade schema safely: inspect existing objects before creating research_jobs."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_pg_types = _inspect_pg_types(bind)
    table_exists = inspector.has_table('research_jobs')

    if not table_exists:
        if 'research_jobs' in existing_pg_types:
            logger.warning(
                "PostgreSQL type 'research_jobs' already exists without table relation; "
                "proceeding cautiously without dropping user data."
            )
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
    else:
        logger.info(
            "Table 'research_jobs' already exists (pg_types=%s); reconciling missing columns/indexes non-destructively.",
            sorted(existing_pg_types),
        )
        existing_cols = {col['name'] for col in inspector.get_columns('research_jobs')}
        for col_name, col_def in _expected_column_definitions(bind.dialect.name):
            if col_name not in existing_cols:
                op.add_column('research_jobs', col_def)

    # Refresh inspector to detect newly created table or existing indexes
    inspector = sa.inspect(bind)
    existing_indexes = {
        idx['name']
        for idx in inspector.get_indexes('research_jobs')
        if idx.get('name')
    }
    for idx_name, idx_cols in _expected_index_definitions():
        if idx_name not in existing_indexes:
            op.create_index(idx_name, 'research_jobs', idx_cols, unique=False)


def downgrade() -> None:
    """Downgrade schema: drop research_jobs table and indexes if present."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table('research_jobs'):
        return

    existing_indexes = {
        idx['name']
        for idx in inspector.get_indexes('research_jobs')
        if idx.get('name')
    }
    for idx_name, _ in reversed(_expected_index_definitions()):
        if idx_name in existing_indexes:
            op.drop_index(idx_name, table_name='research_jobs')
    op.drop_table('research_jobs')
