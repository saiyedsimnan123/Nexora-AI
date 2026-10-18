"""Authentication and authorization application service.

Verifies API credentials against hashed key records in the database, resolves
authenticated user identities, and strictly enforces collection and workspace
ownership on protected operations.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from nexora.api.errors import (
    AuthenticationError,
    DocumentNotFoundError,
    ForbiddenError,
    InvalidConfigurationError,
    InvalidRequestError,
)
from nexora.auth.models import AuthenticatedUser
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.paper import PaperRepository
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository


class AuthenticationService:
    """Service for authenticating callers and enforcing resource authorization."""

    def __init__(self, session_factory: Any) -> None:
        if session_factory is None:
            raise InvalidConfigurationError("AuthenticationService requires a valid session_factory.")
        self._session_factory = session_factory

    def authenticate_token(self, raw_token: str) -> AuthenticatedUser:
        """Authenticate a raw API key token and return the associated AuthenticatedUser.

        Raises:
            AuthenticationError: If token is missing, invalid, revoked, or expired.
        """
        if not isinstance(raw_token, str) or not raw_token.strip():
            raise AuthenticationError("API key token must not be empty.")

        cleaned_token = raw_token.strip()

        with self._session_factory() as session:
            key_repo = ApiKeyRepository(session)
            api_key = key_repo.get_active_by_raw_key(cleaned_token)
            if api_key is None:
                raise AuthenticationError("Invalid or expired API credential.")

            user_repo = UserRepository(session)
            user = user_repo.get_by_id(api_key.user_id)
            if user is None:
                raise AuthenticationError("User associated with this credential no longer exists.")

            # Touch last_used_at timestamp
            key_repo.touch_last_used(api_key.id)
            session.commit()

            return AuthenticatedUser(
                id=user.id,
                email=user.email,
                api_key_id=api_key.id,
                api_key_prefix=api_key.key_prefix,
            )

    def authenticate_request(self, request: Request) -> AuthenticatedUser:
        """Extract and verify credentials from HTTP request headers.

        Supports:
        - `Authorization: Bearer <token>`
        - `X-API-Key: <token>`

        Raises:
            AuthenticationError: If credentials are missing, malformed, or invalid.
        """
        auth_header = request.headers.get("Authorization")
        x_api_key = request.headers.get("X-API-Key")

        if not auth_header and not x_api_key:
            raise AuthenticationError("Authentication credentials were not provided.")

        if auth_header:
            parts = auth_header.strip().split(None, 1)
            if len(parts) != 2 or parts[0].lower() != "bearer":
                raise AuthenticationError(
                    "Invalid authorization scheme. Use 'Authorization: Bearer <token>'."
                )
            raw_token = parts[1].strip()
            if not raw_token:
                raise AuthenticationError("Bearer token value is empty.")
            return self.authenticate_token(raw_token)

        assert x_api_key is not None
        raw_token = x_api_key.strip()
        if not raw_token:
            raise AuthenticationError("X-API-Key header value is empty.")
        return self.authenticate_token(raw_token)

    def verify_collection_ownership(
        self,
        collection_id: UUID | str,
        user_id: UUID,
    ) -> tuple[Collection, Workspace]:
        """Verify that collection_id exists and is owned by user_id through its workspace.

        Raises:
            InvalidRequestError: If collection_id is not a valid UUID.
            DocumentNotFoundError: If collection does not exist in the database (404).
            ForbiddenError: If collection exists but belongs to another user (403).
        """
        try:
            col_uuid = UUID(str(collection_id).strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError("Invalid collection_id: must be a valid UUID.")

        with self._session_factory() as session:
            col_repo = CollectionRepository(session)
            collection = col_repo.get_by_id(col_uuid)
            if collection is None:
                raise DocumentNotFoundError(f"Collection with id {col_uuid} not found.")

            ws_repo = WorkspaceRepository(session)
            workspace = ws_repo.get_by_id(collection.workspace_id)
            if workspace is None or workspace.user_id != user_id:
                raise ForbiddenError(
                    f"Access to collection {col_uuid} is forbidden."
                )

            return collection, workspace

    def verify_paper_ownership(
        self,
        paper_id: UUID | str,
        user_id: UUID,
    ) -> tuple[Paper, Collection, Workspace]:
        """Verify that paper_id exists and is owned by user_id.

        Raises:
            InvalidRequestError: If paper_id is not a valid UUID.
            DocumentNotFoundError: If paper does not exist in the database (404).
            ForbiddenError: If paper belongs to another user's collection (403).
        """
        try:
            paper_uuid = UUID(str(paper_id).strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError("Invalid paper_id: must be a valid UUID.")

        with self._session_factory() as session:
            paper_repo = PaperRepository(session)
            paper = paper_repo.get_by_id(paper_uuid)
            if paper is None:
                raise DocumentNotFoundError(f"Paper with id {paper_uuid} not found.")

            collection, workspace = self.verify_collection_ownership(paper.collection_id, user_id)
            return paper, collection, workspace

    def verify_document_ownership(
        self,
        document_id: UUID | str,
        user_id: UUID,
    ) -> tuple[Document, Paper, Collection, Workspace]:
        """Verify that document_id exists and is owned by user_id.

        Raises:
            InvalidRequestError: If document_id is not a valid UUID.
            DocumentNotFoundError: If document does not exist in the database (404).
            ForbiddenError: If document belongs to another user (403).
        """
        try:
            doc_uuid = UUID(str(document_id).strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError("Invalid document_id: must be a valid UUID.")

        with self._session_factory() as session:
            doc_repo = DocumentRepository(session)
            document = doc_repo.get_by_id(doc_uuid)
            if document is None:
                raise DocumentNotFoundError(f"Document with id {doc_uuid} not found.")

            paper, collection, workspace = self.verify_paper_ownership(document.paper_id, user_id)
            return document, paper, collection, workspace

    def verify_workspace_ownership(
        self,
        workspace_id: UUID | str,
        user_id: UUID,
    ) -> Workspace:
        """Verify that workspace_id exists and is owned by user_id.

        Raises:
            InvalidRequestError: If workspace_id is not a valid UUID.
            DocumentNotFoundError: If workspace does not exist in the database (404).
            ForbiddenError: If workspace belongs to another user (403).
        """
        try:
            ws_uuid = UUID(str(workspace_id).strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError("Invalid workspace_id: must be a valid UUID.")

        with self._session_factory() as session:
            ws_repo = WorkspaceRepository(session)
            workspace = ws_repo.get_by_id(ws_uuid)
            if workspace is None:
                raise DocumentNotFoundError(f"Workspace with id {ws_uuid} not found.")

            if workspace.user_id != user_id:
                raise ForbiddenError(f"Access to workspace {ws_uuid} is forbidden.")

            return workspace
