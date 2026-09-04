#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging

logger = logging.getLogger(__name__)


def resolve_reference_metadata_preferences(
    request_payload: dict | None = None,
    config_payload: dict | None = None,
) -> tuple[bool, set[str] | None]:
    """
    Resolve metadata include/fields from request and optional config.
    Request values take precedence over config values.
    Supports legacy request keys: include_metadata / metadata_fields.
    """
    request_payload = request_payload or {}
    config_payload = config_payload or {}

    config_ref = config_payload.get("reference_metadata", {})
    request_ref = request_payload.get("reference_metadata", {})

    resolved: dict = {}
    if isinstance(config_ref, dict):
        resolved.update(config_ref)
    if isinstance(request_ref, dict):
        resolved.update(request_ref)

    if "include_metadata" in request_payload:
        resolved["include"] = bool(request_payload.get("include_metadata"))
    if "metadata_fields" in request_payload:
        resolved["fields"] = request_payload.get("metadata_fields")

    include_metadata = bool(resolved.get("include", False))
    fields = resolved.get("fields")
    if fields is None:
        return include_metadata, None
    if not isinstance(fields, list):
        logger.warning(
            "reference_metadata.fields is not a list; include_metadata=%s fields=%r type=%s resolved=%r. enrich_chunks_with_document_metadata will skip enrichment.",
            include_metadata,
            fields,
            type(fields).__name__,
            resolved,
        )
        return include_metadata, set()
    return include_metadata, {f for f in fields if isinstance(f, str)}


def enrich_chunks_with_document_metadata(
    chunks: list[dict],
    metadata_fields: set[str] | None = None,
    *,
    kb_field: str = "kb_id",
    doc_field: str = "doc_id",
    output_field: str = "document_metadata",
    include_document_info: bool = False,
) -> None:
    """
    Mutates chunk payloads in-place by attaching `document_metadata`.
    Field names can be customized for different chunk schemas.

    With include_document_info=True the full document record (name, location,
    type, size, chunk_count, create_date, update_date, token_count, thumbnail,
    dataset_id) is attached alongside the custom meta_fields. This matches the
    shape served to the MCP retrieval tool. DB failures are fail-open: the
    enrichment is skipped and the retrieval result is still returned unchanged.
    """
    if metadata_fields is not None and not metadata_fields:
        return

    doc_ids_by_kb: dict[str, set[str]] = {}
    for chunk in chunks:
        kb_ids = chunk.get(kb_field)
        doc_id = chunk.get(doc_field)
        if not kb_ids or not doc_id:
            continue
        if isinstance(kb_ids, (list, tuple)):
            for kid in kb_ids:
                if kid:
                    doc_ids_by_kb.setdefault(kid, set()).add(doc_id)
        else:
            doc_ids_by_kb.setdefault(kb_ids, set()).add(doc_id)

    if not doc_ids_by_kb:
        return

    if include_document_info:
        _attach_full_document_info(chunks, doc_ids_by_kb, metadata_fields, doc_field, output_field)
        return

    # Resolve service lazily so callers/tests that swap service modules at runtime
    # (e.g. via monkeypatch) don't get stuck with a stale class reference.
    from api.db.services.doc_metadata_service import DocMetadataService

    metadata_getter = getattr(DocMetadataService, "get_metadata_for_documents", None)
    if not callable(metadata_getter):
        logging.warning("DocMetadataService.get_metadata_for_documents is unavailable; skipping metadata enrichment.")
        return

    meta_by_doc: dict[str, dict] = {}
    for kb_id, doc_ids in doc_ids_by_kb.items():
        meta_map = metadata_getter(list(doc_ids), kb_id)
        if meta_map:
            meta_by_doc.update(meta_map)
            logging.debug("Fetched metadata for %d docs in kb_id=%s", len(meta_map), kb_id)

    for chunk in chunks:
        doc_id = chunk.get(doc_field)
        if not doc_id:
            continue
        meta = meta_by_doc.get(doc_id)
        if not meta:
            continue
        if metadata_fields is not None:
            meta = {k: v for k, v in meta.items() if k in metadata_fields}
        if meta:
            chunk[output_field] = meta
            logging.debug("Enriched chunk for doc_id=%s with %d metadata fields: %s", doc_id, len(meta), list(meta.keys()))


def _attach_full_document_info(
    chunks: list[dict],
    doc_ids_by_kb: dict[str, set[str]],
    metadata_fields: set[str] | None,
    doc_field: str,
    output_field: str,
) -> None:
    """Attach the full document record + custom metadata for each returned doc.

    The doc records are fetched once per knowledge base via the same batch
    ``doc_ids`` path the REST documents endpoint uses, so only the documents
    referenced by this result page are touched. Any DB failure only disables
    the enrichment (fail-open); retrieval is never blocked by it.
    """
    try:
        from api.db.services.document_service import DocumentService

        doc_getter = getattr(DocumentService, "get_by_kb_id", None)
        if not callable(doc_getter):
            logger.warning("DocumentService.get_by_kb_id is unavailable; skipping document info enrichment.")
            return

        full_by_doc: dict[str, dict] = {}
        for kb_id, doc_ids in doc_ids_by_kb.items():
            docs_list, _ = doc_getter(
                kb_id,
                1,
                len(doc_ids),
                "create_time",
                True,
                None,
                None,
                None,
                None,
                doc_ids=list(doc_ids),
            )
            for doc in docs_list or []:
                if not isinstance(doc, dict):
                    continue
                doc_id = doc.get("id")
                if not doc_id:
                    continue
                meta_fields = doc.get("meta_fields") or {}
                if metadata_fields is not None:
                    meta_fields = {k: v for k, v in meta_fields.items() if k in metadata_fields}
                full_by_doc[doc_id] = {
                    "document_id": doc_id,
                    "name": doc.get("name", ""),
                    "location": doc.get("location", ""),
                    "type": doc.get("type", ""),
                    "size": doc.get("size"),
                    "chunk_count": doc.get("chunk_count", doc.get("chunk_num")),
                    "create_date": doc.get("create_date", ""),
                    "update_date": doc.get("update_date", ""),
                    "token_count": doc.get("token_count", doc.get("token_num")),
                    "thumbnail": doc.get("thumbnail", ""),
                    "dataset_id": doc.get("dataset_id", kb_id),
                    "meta_fields": meta_fields,
                }
            logger.debug("Fetched document info for %d docs in kb_id=%s", len(full_by_doc), kb_id)

        if not full_by_doc:
            return

        for chunk in chunks:
            doc_id = chunk.get(doc_field)
            if not doc_id:
                continue
            meta = full_by_doc.get(doc_id)
            if meta:
                chunk[output_field] = meta
                logger.debug("Enriched chunk for doc_id=%s with full document record", doc_id)
    except Exception:
        logger.exception("Failed to build full document metadata; skipping document info enrichment.")
