"""Tests for the public API of ``nexora.embedding`` (Step 10H).

Import-safety checks run in a fresh subprocess so that the package is
imported for the first time under observation, regardless of what other
tests have already imported.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap

import nexora.embedding as embedding
from nexora.embedding import base, cache, config, factory, hash_provider
from nexora.embedding import pipeline, service

EXPECTED_EXPORTS = {
    "DEFAULT_DIMENSION",
    "EmbeddingCache",
    "EmbeddingConfig",
    "EmbeddingPipeline",
    "EmbeddingProvider",
    "EmbeddingService",
    "HashEmbeddingProvider",
    "InMemoryEmbeddingCache",
    "PROVIDER_HASH",
    "PROVIDER_OPENAI",
    "SUPPORTED_PROVIDERS",
    "create_embedding_provider",
}

SENTINEL_SECRET = "sk-import-test-sentinel-secret"


# 1-2. EmbeddingPipeline
def test_pipeline_importable_from_package():
    from nexora.embedding import EmbeddingPipeline  # noqa: F401


def test_pipeline_is_the_implementation_class():
    assert embedding.EmbeddingPipeline is pipeline.EmbeddingPipeline


# 3. EmbeddingService
def test_service_exported():
    assert embedding.EmbeddingService is service.EmbeddingService


# 4. EmbeddingConfig
def test_config_exported():
    assert embedding.EmbeddingConfig is config.EmbeddingConfig


# 5. EmbeddingProvider
def test_provider_interface_exported():
    assert embedding.EmbeddingProvider is base.EmbeddingProvider


# 6. HashEmbeddingProvider
def test_hash_provider_exported():
    assert embedding.HashEmbeddingProvider is hash_provider.HashEmbeddingProvider


# 7. create_embedding_provider
def test_factory_function_exported():
    assert embedding.create_embedding_provider is factory.create_embedding_provider


# 8. Cache interface and implementation
def test_cache_classes_exported():
    assert embedding.EmbeddingCache is cache.EmbeddingCache
    assert embedding.InMemoryEmbeddingCache is cache.InMemoryEmbeddingCache


# 9. DEFAULT_DIMENSION
def test_default_dimension_exported():
    assert embedding.DEFAULT_DIMENSION == hash_provider.DEFAULT_DIMENSION
    assert isinstance(embedding.DEFAULT_DIMENSION, int)


# 10. Provider constants
def test_provider_constants_exported():
    assert embedding.PROVIDER_HASH == factory.PROVIDER_HASH == "hash"
    assert embedding.PROVIDER_OPENAI == factory.PROVIDER_OPENAI == "openai"
    assert embedding.SUPPORTED_PROVIDERS == factory.SUPPORTED_PROVIDERS
    assert set(embedding.SUPPORTED_PROVIDERS) == {"hash", "openai"}


# Public surface is exactly the intended one
def test_all_matches_expected_exports():
    assert set(embedding.__all__) == EXPECTED_EXPORTS
    assert len(embedding.__all__) == len(set(embedding.__all__))


def test_every_name_in_all_is_resolvable():
    for name in embedding.__all__:
        assert hasattr(embedding, name), name


def test_star_import_exposes_exactly_the_public_api():
    namespace: dict = {}
    exec("from nexora.embedding import *", namespace)
    exported = {k for k in namespace if not k.startswith("__")}
    assert exported == EXPECTED_EXPORTS


# 11-12. Import safety (network and environment access)
_IMPORT_PROBE = textwrap.dedent(
    """
    import json, os, socket

    accessed = []

    class Tracking(dict):
        def get(self, key, default=None):
            accessed.append(str(key))
            return super().get(key, default)

        def __getitem__(self, key):
            accessed.append(str(key))
            return super().__getitem__(key)

        def __contains__(self, key):
            accessed.append(str(key))
            return super().__contains__(key)

    os.environ = Tracking(os.environ)

    network_attempts = []

    def blocked(*args, **kwargs):
        network_attempts.append(1)
        raise RuntimeError("network blocked")

    socket.socket.connect = blocked
    socket.getaddrinfo = blocked
    socket.create_connection = blocked

    import nexora.embedding  # noqa: F401

    print(json.dumps({"accessed": accessed, "network": len(network_attempts)}))
    """
)


def _run_import_probe() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["OPENAI_API_KEY"] = SENTINEL_SECRET
    env["NEXORA_EMBEDDING_API_KEY"] = SENTINEL_SECRET
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert SENTINEL_SECRET not in result.stdout
    assert SENTINEL_SECRET not in result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_import_makes_no_network_request():
    probe = _run_import_probe()
    assert probe["network"] == 0


def test_import_does_not_read_secret_environment_values():
    probe = _run_import_probe()
    sensitive = re.compile(r"OPENAI|NEXORA|API_KEY|SECRET|TOKEN", re.IGNORECASE)
    touched = [key for key in probe["accessed"] if sensitive.search(key)]
    assert touched == []
