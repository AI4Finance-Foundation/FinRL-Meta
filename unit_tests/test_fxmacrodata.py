import json
import threading
from datetime import date
from datetime import timedelta
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.parse import parse_qs
from urllib.parse import urlparse

import pytest

from meta.data_processors import fxmacrodata


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _rows(total):
    # Newest first, like the API.
    start = date(2024, 1, 1)
    return [
        {"date": (start + timedelta(days=i)).isoformat(), "val": float(i)}
        for i in reversed(range(total))
    ]


@pytest.fixture
def fake_api(monkeypatch):
    for name in fxmacrodata.FXMACRODATA_API_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    requests = []
    rows = _rows(250)

    def urlopen(request, timeout=None):
        requests.append(request)
        query = parse_qs(urlparse(request.full_url).query)
        limit = int(query.get("limit", ["20"])[0])
        offset = int(query.get("offset", ["0"])[0])
        page = rows[offset : offset + limit]
        has_more = offset + len(page) < len(rows)
        return _Response(
            {
                "data": page,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "returned_count": len(page),
                    "total_count": len(rows),
                    "has_more": has_more,
                    "next_offset": offset + len(page) if has_more else None,
                },
            }
        )

    monkeypatch.setattr(fxmacrodata, "urlopen", urlopen)
    return requests


def test_api_key_is_sent_as_header(fake_api):
    client = fxmacrodata.FXMacroDataClient(api_key="test-key")
    client.fetch_dataset("forex", base="eur", quote="usd", limit=5)
    request = fake_api[0]
    assert request.get_header("X-api-key") == "test-key"
    assert "test-key" not in request.full_url
    assert "api_key" not in request.full_url


def test_no_api_key_header_without_key(fake_api):
    client = fxmacrodata.FXMacroDataClient()
    client.fetch_dataset("announcements", currency="usd", indicator="inflation")
    request = fake_api[0]
    assert request.get_header("X-api-key") is None
    assert "api_key" not in request.full_url


def test_dataframe_pages_through_window(fake_api):
    client = fxmacrodata.FXMacroDataClient(api_key="test-key")
    frame = client.dataframe(
        "forex", base="eur", quote="usd", start_date="2024-01-01", end_date="2024-12-31"
    )
    assert len(frame) == 250
    assert frame.index.is_monotonic_increasing
    queries = [parse_qs(urlparse(r.full_url).query) for r in fake_api]
    assert [q["offset"] for q in queries] == [["0"], ["100"], ["200"]]
    assert all(q["limit"] == ["100"] for q in queries)
    assert all("api_key" not in r.full_url for r in fake_api)


def test_dataframe_limit_caps_rows_and_page_size(fake_api):
    client = fxmacrodata.FXMacroDataClient()
    frame = client.dataframe("cot", currency="eur", limit=150)
    assert len(frame) == 150
    queries = [parse_qs(urlparse(r.full_url).query) for r in fake_api]
    assert [(q["limit"], q["offset"]) for q in queries] == [
        (["100"], ["0"]),
        (["50"], ["100"]),
    ]
    assert frame.index.max().isoformat().startswith("2024-09-06")


def test_fetch_dataset_never_sends_limit_above_100(fake_api):
    client = fxmacrodata.FXMacroDataClient()
    client.fetch_dataset("commodity", indicator="gold", limit=500)
    assert parse_qs(urlparse(fake_api[0].full_url).query)["limit"] == ["100"]


def test_redirect_is_not_followed(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            if self.path.startswith("/v1/"):
                self.send_response(302)
                self.send_header(
                    "Location", "http://127.0.0.1:%d/target" % self.server.server_port
                )
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = fxmacrodata.FXMacroDataClient(
            api_key="secret-key",
            base_url="http://127.0.0.1:%d/v1" % server.server_port,
        )
        with pytest.raises(HTTPError) as excinfo:
            client.fetch_dataset("forex", base="eur", quote="usd")
    finally:
        server.shutdown()
        server.server_close()

    assert excinfo.value.code == 302
    assert "secret-key" not in str(excinfo.value)
    assert paths == ["/v1/forex/eur/usd"]


@pytest.mark.parametrize("api_key", ["secret\nkey", "secret key", "secret\x00key"])
def test_bad_api_key_is_rejected_without_echoing_it(fake_api, api_key):
    client = fxmacrodata.FXMacroDataClient(api_key=api_key)
    with pytest.raises(ValueError) as excinfo:
        client.fetch_dataset("forex", base="eur", quote="usd")
    assert "invalid characters" in str(excinfo.value)
    assert "secret" not in str(excinfo.value)
    assert fake_api == []


def test_api_key_is_stripped(fake_api):
    client = fxmacrodata.FXMacroDataClient(api_key=" test-key\n")
    client.fetch_dataset("forex", base="eur", quote="usd", limit=5)
    assert fake_api[0].get_header("X-api-key") == "test-key"


@pytest.mark.parametrize(
    "body, message",
    [
        (json.dumps({"detail": "Invalid API key"}).encode(), "Invalid API key"),
        (b"<html>Service Unavailable</html>", "non-JSON response"),
        (b'"unexpected"', "expected a JSON object"),
    ],
)
def test_fetch_dataset_rejects_error_bodies(monkeypatch, body, message):
    def urlopen(request, timeout=None):
        response = _Response(None)
        response._body = body
        return response

    monkeypatch.setattr(fxmacrodata, "urlopen", urlopen)
    client = fxmacrodata.FXMacroDataClient()
    with pytest.raises(ValueError, match=message):
        client.fetch_dataset("calendar", currency="usd")


def test_fetch_rows_rejects_data_that_is_not_a_list(monkeypatch):
    monkeypatch.setattr(
        fxmacrodata,
        "urlopen",
        lambda request, timeout=None: _Response({"data": {"date": "2024-01-01"}}),
    )
    client = fxmacrodata.FXMacroDataClient()
    with pytest.raises(ValueError, match="expected a JSON object"):
        client.fetch_rows("forex", base="eur", quote="usd")


def test_fetch_rows_skips_malformed_rows(monkeypatch):
    monkeypatch.setattr(
        fxmacrodata,
        "urlopen",
        lambda request, timeout=None: _Response(
            {"data": ["2024-01-02", None, {"date": "2024-01-01", "val": 1.1}]}
        ),
    )
    client = fxmacrodata.FXMacroDataClient()
    assert client.fetch_rows("forex", base="eur", quote="usd") == [
        {"date": "2024-01-01", "val": 1.1}
    ]


@pytest.mark.parametrize("next_offset", [0, "1"])
def test_fetch_rows_stops_when_offset_does_not_advance(monkeypatch, next_offset):
    requests = []

    def urlopen(request, timeout=None):
        requests.append(request)
        return _Response(
            {
                "data": [{"date": "2024-01-01", "val": 1.1}],
                "pagination": {"has_more": True, "next_offset": next_offset},
            }
        )

    monkeypatch.setattr(fxmacrodata, "urlopen", urlopen)
    client = fxmacrodata.FXMacroDataClient()
    rows = client.fetch_rows("forex", base="eur", quote="usd")
    assert len(requests) == 1
    assert rows == [{"date": "2024-01-01", "val": 1.1}]
