"""Search and PII-Filter settings seed

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-26

"""

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    settings = [
        # Search Settings (SearXNG)
        (
            "SEARCH_SEARXNG_URL",
            "http://searxng:8080",
            "search",
            False,
            "SearXNG Metasuchmaschine URL (z.B. http://localhost:8080)",
        ),
        (
            "SEARCH_SEARXNG_CATEGORIES",
            "general",
            "search",
            False,
            "Durchsuchbare Kategorien (Komma-getrennt, z.B. general,news,science)",
        ),
        (
            "SEARCH_SEARXNG_ENABLED",
            "true",
            "search",
            False,
            "Websuche aktivieren (true/false)",
        ),
        # PII-Filter Settings
        (
            "PII_FILTER_URL",
            "http://entityguard:9500/api/v1/sanitize",
            "pii_filter",
            False,
            "Vollständige PII-Filter Endpoint URL (z.B. http://localhost:9500/api/v1/sanitize)",
        ),
        (
            "PII_FILTER_API_KEY",
            "",
            "pii_filter",
            True,
            "API Key für PII-Filter Service",
        ),
        (
            "PII_FILTER_ENABLED",
            "true",
            "pii_filter",
            False,
            "PII-Filter aktivieren (true/false)",
        ),
    ]

    for key, value, category, is_sensitive, description in settings:
        conn.execute(
            sa.text(
                "INSERT INTO settings "
                "(key, value, default_value, category, is_sensitive, description) "
                "VALUES (:key, :value, :value, :category, :is_sensitive, :description) "
                "ON CONFLICT (key) DO NOTHING"
            ),
            {
                "key": key,
                "value": value,
                "category": category,
                "is_sensitive": is_sensitive,
                "description": description,
            },
        )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "DELETE FROM settings WHERE key IN "
            "('SEARCH_SEARXNG_URL', 'SEARCH_SEARXNG_CATEGORIES', 'SEARCH_SEARXNG_ENABLED', "
            "'PII_FILTER_URL', 'PII_FILTER_API_KEY', 'PII_FILTER_ENABLED')"
        )
    )
