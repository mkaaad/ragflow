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

import importlib.util
import json
from collections import OrderedDict
from pathlib import Path

import pytest


def _load_mcp_server():
    server_path = Path(__file__).resolve().parents[3] / "mcp" / "server" / "server.py"
    spec = importlib.util.spec_from_file_location("ragflow_mcp_server_unit", server_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture()
def mcp_server():
    return _load_mcp_server()


def _fresh_connector(mcp_server):
    connector = mcp_server.RAGFlowConnector(base_url=mcp_server.BASE_URL)
    # The dataset metadata cache is a class-level OrderedDict shared across
    # instances; shadow it per test so cases don't pollute one another.
    connector._dataset_metadata_cache = OrderedDict()
    return connector


def _retrieval_response(chunks, total=1):
    return _FakeResponse(
        {
            "code": 0,
            "data": {
                "total": total,
                "chunks": chunks,
                "page": 1,
                "page_size": 30,
            },
        }
    )


@pytest.mark.asyncio
async def test_retrieval_requests_document_metadata_from_backend(monkeypatch, mcp_server):
    connector = _fresh_connector(mcp_server)
    captured = {}

    async def _post(path, json=None, stream=False, files=None, api_key=""):
        captured["body"] = json
        return _retrieval_response([{"id": "c1", "content": "x", "document_id": "d1", "dataset_id": "kb-1"}])

    async def _get(path, params=None, api_key=""):
        return _FakeResponse({"code": 0, "data": []})

    monkeypatch.setattr(connector, "_post", _post)
    monkeypatch.setattr(connector, "_get", _get)

    result = await connector.retrieval(api_key="k", dataset_ids=["kb-1"], question="q")

    assert json.loads(result[0].text)["chunks"][0]["id"] == "c1"
    assert captured["body"]["reference_metadata"] == {"include": True, "include_document_info": True}


@pytest.mark.asyncio
async def test_retrieval_passes_through_backend_document_metadata(monkeypatch, mcp_server):
    connector = _fresh_connector(mcp_server)
    doc_meta = {
        "document_id": "d1",
        "name": "doc.pdf",
        "location": "loc",
        "type": "pdf",
        "size": 100,
        "chunk_count": 3,
        "create_date": "2026-01-01",
        "update_date": "2026-01-02",
        "token_count": 50,
        "thumbnail": "",
        "dataset_id": "kb-1",
        "meta_fields": {"author": "x"},
    }

    async def _post(path, json=None, stream=False, files=None, api_key=""):
        return _retrieval_response(
            [
                {
                    "id": "c1",
                    "content": "x",
                    "document_id": "d1",
                    "dataset_id": "kb-1",
                    "document_keyword": "doc.pdf",
                    "document_metadata": doc_meta,
                }
            ]
        )

    async def _get(path, params=None, api_key=""):
        if (params or {}).get("id") == "kb-1":
            return _FakeResponse({"code": 0, "data": [{"name": "MyKB", "description": "d"}]})
        return _FakeResponse({"code": 0, "data": []})

    monkeypatch.setattr(connector, "_post", _post)
    monkeypatch.setattr(connector, "_get", _get)

    result = await connector.retrieval(api_key="k", dataset_ids=["kb-1"], question="q")

    chunk = json.loads(result[0].text)["chunks"][0]
    assert chunk["dataset_name"] == "MyKB"
    assert chunk["document_name"] == "doc.pdf"
    assert chunk["document_metadata"] == doc_meta


def test_map_chunk_fields_passes_through_document_metadata(mcp_server):
    connector = _fresh_connector(mcp_server)
    chunk = {
        "id": "c1",
        "content": "x",
        "document_id": "d1",
        "dataset_id": "kb-1",
        "document_keyword": "doc.pdf",
        "document_metadata": {"document_id": "d1", "name": "doc.pdf"},
    }

    mapped = connector._map_chunk_fields(chunk, {"kb-1": {"name": "MyKB", "description": "d"}})

    assert mapped["dataset_name"] == "MyKB"
    assert mapped["document_name"] == "doc.pdf"
    assert mapped["document_metadata"] == {"document_id": "d1", "name": "doc.pdf"}


def test_map_chunk_fields_without_document_metadata(mcp_server):
    connector = _fresh_connector(mcp_server)
    chunk = {"id": "c1", "content": "x", "document_id": "d1", "dataset_id": "kb-1", "document_keyword": "doc.pdf"}

    mapped = connector._map_chunk_fields(chunk, {})

    assert "document_metadata" not in mapped
    assert mapped["dataset_name"] == "Unknown"
    assert mapped["document_name"] == "doc.pdf"


@pytest.mark.asyncio
async def test_dataset_metadata_continues_after_failing_dataset(monkeypatch, mcp_server):
    connector = _fresh_connector(mcp_server)

    async def _get(path, params=None, api_key=""):
        dataset_id = (params or {}).get("id")
        if dataset_id == "bad-ds":
            raise RuntimeError("boom")
        if dataset_id == "good-ds":
            return _FakeResponse({"code": 0, "data": [{"name": "GoodKB", "description": "d"}]})
        return _FakeResponse({"code": 0, "data": []})

    monkeypatch.setattr(connector, "_get", _get)

    cache = await connector._get_dataset_metadata_cache(["bad-ds", "good-ds"], api_key="k")

    assert "bad-ds" not in cache
    assert cache["good-ds"]["name"] == "GoodKB"


@pytest.mark.asyncio
async def test_dataset_metadata_cache_reuses_cached_entries(monkeypatch, mcp_server):
    connector = _fresh_connector(mcp_server)
    calls = []

    async def _get(path, params=None, api_key=""):
        calls.append((path, (params or {}).get("id")))
        return _FakeResponse({"code": 0, "data": [{"name": "KB", "description": "d"}]})

    monkeypatch.setattr(connector, "_get", _get)

    first = await connector._get_dataset_metadata_cache(["kb-1"], api_key="k")
    second = await connector._get_dataset_metadata_cache(["kb-1"], api_key="k")

    assert first == second == {"kb-1": {"name": "KB", "description": "d"}}
    assert [c for c in calls if c[0] == "/datasets"] == [("/datasets", "kb-1")]
