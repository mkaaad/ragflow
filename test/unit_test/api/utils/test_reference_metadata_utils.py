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

from api.utils.reference_metadata_utils import (
    enrich_chunks_with_document_metadata,
    resolve_reference_metadata_preferences,
)


def test_resolve_defaults_to_not_enriching():
    assert resolve_reference_metadata_preferences({}) == (False, None)
    assert resolve_reference_metadata_preferences(None, None) == (False, None)


def test_resolve_include_document_info_is_not_part_of_shared_contract():
    # The shared resolver keeps its 2-tuple contract; include_document_info is
    # consumed locally by the /retrieval handler instead.
    assert resolve_reference_metadata_preferences({"reference_metadata": {"include": True, "include_document_info": True}}) == (True, None)


def test_resolve_legacy_include_metadata_and_fields():
    assert resolve_reference_metadata_preferences({"include_metadata": True, "metadata_fields": ["author"]}) == (True, {"author"})


def test_enrich_full_document_info_attaches_curated_record(monkeypatch):
    chunks = [
        {"kb_id": "kb-1", "doc_id": "doc-1", "content": "a"},
        {"kb_id": "kb-1", "doc_id": "doc-2", "content": "b"},
        {"kb_id": "kb-1", "doc_id": "doc-3", "content": "c"},  # not present in DB
    ]

    def fake_get_by_kb_id(kb_id, page_number, items_per_page, orderby, desc, keywords, run_status, types, suffix, name=None, doc_ids=None, return_empty_metadata=False):
        assert kb_id == "kb-1"
        assert set(doc_ids) == {"doc-1", "doc-2", "doc-3"}
        return [
            {
                "id": "doc-1",
                "name": "a.pdf",
                "location": "loc/a",
                "type": "pdf",
                "size": 100,
                "chunk_num": 3,
                "token_num": 50,
                "create_date": "2026-01-01",
                "update_date": "2026-01-02",
                "thumbnail": "thumb-a",
                "kb_id": "kb-1",
                "meta_fields": {"author": "x"},
            },
            {
                "id": "doc-2",
                "name": "b.pdf",
                "location": "",
                "type": "pdf",
                "size": 200,
                "chunk_num": 5,
                "token_num": 80,
                "create_date": "2026-02-01",
                "update_date": "",
                "thumbnail": "",
                "kb_id": "kb-1",
                "meta_fields": {},
            },
        ], 2

    monkeypatch.setattr("api.db.services.document_service.DocumentService.get_by_kb_id", fake_get_by_kb_id)

    enrich_chunks_with_document_metadata(chunks, None, include_document_info=True)

    assert chunks[0]["document_metadata"] == {
        "document_id": "doc-1",
        "name": "a.pdf",
        "location": "loc/a",
        "type": "pdf",
        "size": 100,
        "chunk_count": 3,
        "create_date": "2026-01-01",
        "update_date": "2026-01-02",
        "token_count": 50,
        "thumbnail": "thumb-a",
        "dataset_id": "kb-1",
        "meta_fields": {"author": "x"},
    }
    assert chunks[1]["document_metadata"]["document_id"] == "doc-2"
    assert "document_metadata" not in chunks[2]


def test_enrich_full_document_info_filters_meta_fields(monkeypatch):
    chunks = [{"kb_id": "kb-1", "doc_id": "doc-1", "content": "a"}]

    monkeypatch.setattr(
        "api.db.services.document_service.DocumentService.get_by_kb_id",
        lambda *a, **k: ([{"id": "doc-1", "name": "a", "kb_id": "kb-1", "meta_fields": {"author": "x", "status": "done"}}], 1),
    )

    enrich_chunks_with_document_metadata(chunks, {"status"}, include_document_info=True)

    assert chunks[0]["document_metadata"]["meta_fields"] == {"status": "done"}


def test_enrich_full_document_info_fails_open(monkeypatch):
    chunks = [{"kb_id": "kb-1", "doc_id": "doc-1", "content": "a"}]

    def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr("api.db.services.document_service.DocumentService.get_by_kb_id", boom)

    # Must not raise; enrichment is skipped and chunks stay untouched.
    enrich_chunks_with_document_metadata(chunks, None, include_document_info=True)
    assert "document_metadata" not in chunks[0]


def test_enrich_custom_metadata_only_by_default(monkeypatch):
    chunks = [{"kb_id": "kb-1", "doc_id": "doc-1", "content": "a"}]

    monkeypatch.setattr(
        "api.db.services.doc_metadata_service.DocMetadataService.get_metadata_for_documents",
        lambda doc_ids, kb_id: {"doc-1": {"author": "x"}},
    )

    enrich_chunks_with_document_metadata(chunks, None)
    assert chunks[0]["document_metadata"] == {"author": "x"}
