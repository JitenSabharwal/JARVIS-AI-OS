from types import SimpleNamespace

from interfaces.api_interface import APIInterface


def _req(path: str, *, auth: str = "", query: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        headers={"Authorization": auth} if auth else {},
        query=query or {},
    )


def test_extract_request_token_prefers_authorization_header() -> None:
    api = APIInterface(auth_token="secret")
    req = _req(
        "/api/v1/realtime/sessions/s1/ws",
        auth="Bearer header-token",
        query={"access_token": "query-token"},
    )
    assert api._extract_request_token(req) == "header-token"


def test_extract_request_token_allows_ws_query_token() -> None:
    api = APIInterface(auth_token="secret")
    req = _req(
        "/api/v1/realtime/sessions/s1/ws",
        query={"access_token": "query-token"},
    )
    assert api._extract_request_token(req) == "query-token"


def test_extract_request_token_blocks_query_token_on_non_ws_routes() -> None:
    api = APIInterface(auth_token="secret")
    req = _req("/api/v1/query", query={"access_token": "query-token"})
    assert api._extract_request_token(req) == ""


def test_infer_streaming_hints_from_query_sections() -> None:
    hints = APIInterface._infer_streaming_hints_from_query("yes do it in 4 sections and stream")
    assert hints.get("sectioned") is True
    assert hints.get("max_sections") == 4


def test_infer_streaming_hints_from_query_empty_when_no_signal() -> None:
    hints = APIInterface._infer_streaming_hints_from_query("explain react")
    assert hints == {}
