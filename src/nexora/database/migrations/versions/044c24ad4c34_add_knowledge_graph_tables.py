"""Add knowledge graph entities and relationships tables (Stage 5 Step 1).

Revision ID: 044c24ad4c34
Revises: 044c24ad4c33
Create Date: 2026-10-25 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '044c24ad4c34'
down_revision: Union[str, Sequence[str], None] = '044c24ad4c33'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add graph_entities and graph_relationships tables."""
    # 1. Create graph_entities table
    op.create_table(
        'graph_entities',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('collection_id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('canonical_name', sa.String(length=255), nullable=False),
        sa.Column('entity_type', sa.String(length=50), nullable=False),
        sa.Column('description', sa.String(length=5000), nullable=True),
        sa.Column('aliases', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('entity_metadata', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], name=op.f('fk_graph_entities_collection_id_collections'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_graph_entities')),
        sa.UniqueConstraint('collection_id', 'entity_type', 'canonical_name', name=op.f('uq_graph_entities_collection_id')),
        sa.CheckConstraint('length(trim(name)) > 0', name=op.f('ck_graph_entities_name_not_blank')),
        sa.CheckConstraint('length(trim(canonical_name)) > 0', name=op.f('ck_graph_entities_canonical_name_not_blank')),
        sa.CheckConstraint("entity_type IN ('concept', 'method', 'dataset', 'metric', 'author', 'other')", name=op.f('ck_graph_entities_valid_entity_type')),
    )
    op.create_index(op.f('ix_graph_entities_collection_id'), 'graph_entities', ['collection_id'], unique=False)
    op.create_index(op.f('ix_graph_entities_canonical_name'), 'graph_entities', ['canonical_name'], unique=False)
    op.create_index(op.f('ix_graph_entities_entity_type'), 'graph_entities', ['entity_type'], unique=False)

    # 2. Create graph_relationships table
    op.create_table(
        'graph_relationships',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('collection_id', sa.Uuid(), nullable=False),
        sa.Column('relationship_type', sa.String(length=100), nullable=False),
        sa.Column('source_type', sa.String(length=20), nullable=False),
        sa.Column('source_id', sa.Uuid(), nullable=False),
        sa.Column('target_type', sa.String(length=20), nullable=False),
        sa.Column('target_id', sa.Uuid(), nullable=False),
        sa.Column('source_paper_id', sa.Uuid(), nullable=True),
        sa.Column('source_entity_id', sa.Uuid(), nullable=True),
        sa.Column('target_paper_id', sa.Uuid(), nullable=True),
        sa.Column('target_entity_id', sa.Uuid(), nullable=True),
        sa.Column('paper_id', sa.Uuid(), nullable=True),
        sa.Column('chunk_id', sa.Uuid(), nullable=True),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('evidence_text', sa.String(length=5000), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('extraction_method', sa.String(length=100), nullable=True),
        sa.Column('relationship_metadata', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], name=op.f('fk_graph_relationships_collection_id_collections'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['source_paper_id'], ['papers.id'], name=op.f('fk_graph_relationships_source_paper_id_papers'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['source_entity_id'], ['graph_entities.id'], name=op.f('fk_graph_relationships_source_entity_id_graph_entities'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_paper_id'], ['papers.id'], name=op.f('fk_graph_relationships_target_paper_id_papers'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_entity_id'], ['graph_entities.id'], name=op.f('fk_graph_relationships_target_entity_id_graph_entities'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], name=op.f('fk_graph_relationships_paper_id_papers'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['chunk_id'], ['document_chunks.id'], name=op.f('fk_graph_relationships_chunk_id_document_chunks'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_graph_relationships')),
        sa.UniqueConstraint('collection_id', 'relationship_type', 'source_type', 'source_id', 'target_type', 'target_id', name=op.f('uq_graph_relationships_identity')),
        sa.CheckConstraint('length(trim(relationship_type)) > 0', name=op.f('ck_graph_relationships_type_not_blank')),
        sa.CheckConstraint("source_type IN ('paper', 'entity')", name=op.f('ck_graph_relationships_source_type_valid')),
        sa.CheckConstraint("target_type IN ('paper', 'entity')", name=op.f('ck_graph_relationships_target_type_valid')),
        sa.CheckConstraint('NOT (source_type = target_type AND source_id = target_id)', name=op.f('ck_graph_relationships_no_self_loop')),
        sa.CheckConstraint('confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)', name=op.f('ck_graph_relationships_confidence_range')),
        sa.CheckConstraint('page_number IS NULL OR page_number >= 1', name=op.f('ck_graph_relationships_page_number_positive')),
        sa.CheckConstraint('evidence_text IS NULL OR length(trim(evidence_text)) > 0', name=op.f('ck_graph_relationships_evidence_not_blank')),
        sa.CheckConstraint("((source_type = 'paper' AND source_paper_id IS NOT NULL AND source_paper_id = source_id AND source_entity_id IS NULL) OR (source_type = 'entity' AND source_entity_id IS NOT NULL AND source_entity_id = source_id AND source_paper_id IS NULL))", name=op.f('ck_graph_relationships_source_endpoint_consistency')),
        sa.CheckConstraint("((target_type = 'paper' AND target_paper_id IS NOT NULL AND target_paper_id = target_id AND target_entity_id IS NULL) OR (target_type = 'entity' AND target_entity_id IS NOT NULL AND target_entity_id = target_id AND target_paper_id IS NULL))", name=op.f('ck_graph_relationships_target_endpoint_consistency')),
    )
    op.create_index(op.f('ix_graph_relationships_collection_id'), 'graph_relationships', ['collection_id'], unique=False)
    op.create_index(op.f('ix_graph_relationships_relationship_type'), 'graph_relationships', ['relationship_type'], unique=False)
    op.create_index(op.f('ix_graph_relationships_source_id'), 'graph_relationships', ['source_id'], unique=False)
    op.create_index(op.f('ix_graph_relationships_target_id'), 'graph_relationships', ['target_id'], unique=False)
    op.create_index(op.f('ix_graph_relationships_source_paper_id'), 'graph_relationships', ['source_paper_id'], unique=False)
    op.create_index(op.f('ix_graph_relationships_source_entity_id'), 'graph_relationships', ['source_entity_id'], unique=False)
    op.create_index(op.f('ix_graph_relationships_target_paper_id'), 'graph_relationships', ['target_paper_id'], unique=False)
    op.create_index(op.f('ix_graph_relationships_target_entity_id'), 'graph_relationships', ['target_entity_id'], unique=False)
    op.create_index(op.f('ix_graph_relationships_paper_id'), 'graph_relationships', ['paper_id'], unique=False)
    op.create_index(op.f('ix_graph_relationships_chunk_id'), 'graph_relationships', ['chunk_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema: drop graph_relationships and graph_entities tables."""
    op.drop_index(op.f('ix_graph_relationships_chunk_id'), table_name='graph_relationships')
    op.drop_index(op.f('ix_graph_relationships_paper_id'), table_name='graph_relationships')
    op.drop_index(op.f('ix_graph_relationships_target_entity_id'), table_name='graph_relationships')
    op.drop_index(op.f('ix_graph_relationships_target_paper_id'), table_name='graph_relationships')
    op.drop_index(op.f('ix_graph_relationships_source_entity_id'), table_name='graph_relationships')
    op.drop_index(op.f('ix_graph_relationships_source_paper_id'), table_name='graph_relationships')
    op.drop_index(op.f('ix_graph_relationships_target_id'), table_name='graph_relationships')
    op.drop_index(op.f('ix_graph_relationships_source_id'), table_name='graph_relationships')
    op.drop_index(op.f('ix_graph_relationships_relationship_type'), table_name='graph_relationships')
    op.drop_index(op.f('ix_graph_relationships_collection_id'), table_name='graph_relationships')
    op.drop_table('graph_relationships')

    op.drop_index(op.f('ix_graph_entities_entity_type'), table_name='graph_entities')
    op.drop_index(op.f('ix_graph_entities_canonical_name'), table_name='graph_entities')
    op.drop_index(op.f('ix_graph_entities_collection_id'), table_name='graph_entities')
    op.drop_table('graph_entities')
