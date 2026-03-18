"""
Web search and HTTP fetch skills for JARVIS AI OS.

All skills use only standard-library or widely-available packages:

- :class:`WebSearchSkill`  – DuckDuckGo Instant Answer API (no key required)
- :class:`URLFetchSkill`   – Fetch arbitrary URL content
- :class:`WeatherSkill`    – Current weather via wttr.in (no key required)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from infrastructure.logger import get_logger
from skills.base_skill import BaseSkill, SkillParameter, SkillResult

logger = get_logger(__name__)

# Optional: use requests when available for richer response handling.
try:
    import requests as _requests  # type: ignore

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_DEFAULT_TIMEOUT = 15  # seconds
_USER_AGENT = (
    "Mozilla/5.0 (compatible; JARVIS-AI-OS/1.0; +https://github.com/JARVIS-AI-OS)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: int = _DEFAULT_TIMEOUT) -> tuple[int, str]:
    """Perform an HTTP GET and return ``(status_code, body_text)``.

    Uses *requests* when available, falls back to *urllib*.
    """
    if _REQUESTS_AVAILABLE:
        resp = _requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
        return resp.status_code, resp.text
    else:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read()
                charset = response.headers.get_content_charset("utf-8")
                return response.status, body.decode(charset, errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, str(exc.reason)


# ---------------------------------------------------------------------------
# WebSearchSkill
# ---------------------------------------------------------------------------


class WebSearchSkill(BaseSkill):
    """Search the web using the DuckDuckGo Instant Answer API.

    The DuckDuckGo API (``https://api.duckduckgo.com/``) is free, requires no
    API key, and returns JSON with an abstract, related topics, and more.
    """

    _DDG_URL = "https://api.duckduckgo.com/"

    max_retries: int = 2
    retry_delay: float = 1.0

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information using the DuckDuckGo Instant Answer API. "
            "Returns abstract text, related topics, and source URLs."
        )

    @property
    def category(self) -> str:
        return "web"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                name="query",
                type="string",
                description="The search query to look up.",
                required=True,
            ),
            SkillParameter(
                name="max_results",
                type="integer",
                description="Maximum number of related topics to return (1–20).",
                required=False,
                default=5,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not params.get("query", "").strip():
            raise ValueError("'query' must be a non-empty string.")
        max_results = params.get("max_results", 5)
        if not isinstance(max_results, int) or not (1 <= max_results <= 20):
            raise ValueError("'max_results' must be an integer between 1 and 20.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        query = params["query"].strip()
        max_results = int(params.get("max_results", 5))

        encoded = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"{self._DDG_URL}?{encoded}"

        try:
            status, body = _http_get(url)
        except Exception as exc:
            return SkillResult.failure(
                error=f"HTTP request failed: {exc}",
                metadata={"url": url},
            )

        if status != 200:
            return SkillResult.failure(
                error=f"DuckDuckGo API returned HTTP {status}.",
                metadata={"url": url, "status_code": status},
            )

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            return SkillResult.failure(
                error=f"Failed to parse API response: {exc}",
                metadata={"url": url},
            )

        abstract = data.get("AbstractText", "")
        abstract_source = data.get("AbstractSource", "")
        abstract_url = data.get("AbstractURL", "")

        related_topics = []
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                related_topics.append({
                    "text": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                })

        result_data = {
            "query": query,
            "abstract": abstract,
            "abstract_source": abstract_source,
            "abstract_url": abstract_url,
            "related_topics": related_topics,
            "definition": data.get("Definition", ""),
            "definition_source": data.get("DefinitionSource", ""),
            "answer": data.get("Answer", ""),
            "answer_type": data.get("AnswerType", ""),
        }

        if not abstract and not related_topics and not data.get("Answer"):
            return SkillResult(
                success=True,
                data=result_data,
                metadata={"note": "No results found for query.", "url": url},
            )

        return SkillResult.ok(data=result_data, metadata={"url": url, "status_code": status})


# ---------------------------------------------------------------------------
# URLFetchSkill
# ---------------------------------------------------------------------------


class URLFetchSkill(BaseSkill):
    """Fetch the content of any URL and return it as text.

    Handles redirects automatically.  Use *extract_text* to strip HTML tags
    and return only visible text.
    """

    @property
    def name(self) -> str:
        return "url_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch the raw or text content of a URL. "
            "Can optionally strip HTML tags to return readable text."
        )

    @property
    def category(self) -> str:
        return "web"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                name="url",
                type="string",
                description="The URL to fetch (must start with http:// or https://).",
                required=True,
            ),
            SkillParameter(
                name="extract_text",
                type="boolean",
                description="Strip HTML tags and return only visible text.",
                required=False,
                default=False,
            ),
            SkillParameter(
                name="max_chars",
                type="integer",
                description="Truncate response body to this many characters (0 = unlimited).",
                required=False,
                default=10000,
            ),
            SkillParameter(
                name="timeout",
                type="integer",
                description="Request timeout in seconds (1–60).",
                required=False,
                default=15,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        url = params.get("url", "")
        if not url.startswith(("http://", "https://")):
            raise ValueError("'url' must start with http:// or https://")
        timeout = params.get("timeout", 15)
        if not isinstance(timeout, int) or not (1 <= timeout <= 60):
            raise ValueError("'timeout' must be an integer between 1 and 60.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        url = params["url"]
        extract_text = bool(params.get("extract_text", False))
        max_chars = int(params.get("max_chars", 10000))
        timeout = int(params.get("timeout", 15))

        try:
            status, body = _http_get(url, timeout=timeout)
        except Exception as exc:
            return SkillResult.failure(
                error=f"Request failed: {exc}",
                metadata={"url": url},
            )

        if status >= 400:
            return SkillResult.failure(
                error=f"HTTP error {status} fetching URL.",
                metadata={"url": url, "status_code": status},
            )

        if extract_text:
            body = self._strip_html(body)

        if max_chars and len(body) > max_chars:
            body = body[:max_chars]

        return SkillResult.ok(
            data={"url": url, "content": body, "length": len(body)},
            metadata={"status_code": status, "truncated": max_chars > 0 and len(body) >= max_chars},
        )

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and collapse whitespace."""
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# ---------------------------------------------------------------------------
# WeatherSkill
# ---------------------------------------------------------------------------


class WeatherSkill(BaseSkill):
    """Retrieve current weather data for any location using wttr.in.

    ``wttr.in`` is a free, no-key-required weather service that returns
    structured JSON data.
    """

    _WTTR_URL = "https://wttr.in/{location}?format=j1"

    max_retries: int = 2
    retry_delay: float = 2.0

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return (
            "Get current weather conditions and forecast for a location. "
            "Uses the wttr.in free weather API (no API key required)."
        )

    @property
    def category(self) -> str:
        return "web"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                name="location",
                type="string",
                description=(
                    "City name, address, or coordinates (e.g. 'New York', "
                    "'London', '48.8566,2.3522')."
                ),
                required=True,
            ),
            SkillParameter(
                name="units",
                type="string",
                description="Temperature units: 'metric' (Celsius) or 'imperial' (Fahrenheit).",
                required=False,
                default="metric",
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        location = params.get("location", "").strip()
        if not location:
            raise ValueError("'location' must be a non-empty string.")
        units = params.get("units", "metric")
        if units not in ("metric", "imperial"):
            raise ValueError("'units' must be 'metric' or 'imperial'.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        location = params["location"].strip()
        units = params.get("units", "metric")
        use_fahrenheit = units == "imperial"

        encoded_location = urllib.parse.quote(location)
        url = self._WTTR_URL.format(location=encoded_location)

        try:
            status, body = _http_get(url)
        except Exception as exc:
            return SkillResult.failure(
                error=f"Request to wttr.in failed: {exc}",
                metadata={"location": location},
            )

        if status != 200:
            return SkillResult.failure(
                error=f"Weather API returned HTTP {status}.",
                metadata={"location": location, "status_code": status},
            )

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            return SkillResult.failure(
                error=f"Failed to parse weather data: {exc}",
                metadata={"location": location},
            )

        try:
            current = data["current_condition"][0]
            area = data.get("nearest_area", [{}])[0]
            area_name = (
                area.get("areaName", [{}])[0].get("value", location)
                if area.get("areaName")
                else location
            )
            country = (
                area.get("country", [{}])[0].get("value", "")
                if area.get("country")
                else ""
            )

            temp_c = float(current.get("temp_C", 0))
            temp_f = float(current.get("temp_F", temp_c * 9 / 5 + 32))
            feels_like_c = float(current.get("FeelsLikeC", temp_c))
            feels_like_f = float(current.get("FeelsLikeF", feels_like_c * 9 / 5 + 32))
            humidity = current.get("humidity", "N/A")
            wind_kmph = current.get("windspeedKmph", "N/A")
            wind_dir = current.get("winddir16Point", "N/A")
            visibility_km = current.get("visibility", "N/A")
            description = (
                current.get("weatherDesc", [{}])[0].get("value", "Unknown")
                if current.get("weatherDesc")
                else "Unknown"
            )
            uv_index = current.get("uvIndex", "N/A")

            if use_fahrenheit:
                temp_display = f"{temp_f:.1f}°F"
                feels_like_display = f"{feels_like_f:.1f}°F"
            else:
                temp_display = f"{temp_c:.1f}°C"
                feels_like_display = f"{feels_like_c:.1f}°C"

            # 3-day forecast
            forecast = []
            for day in data.get("weather", []):
                date = day.get("date", "")
                max_c = float(day.get("maxtempC", 0))
                min_c = float(day.get("mintempC", 0))
                max_f = float(day.get("maxtempF", max_c * 9 / 5 + 32))
                min_f = float(day.get("mintempF", min_c * 9 / 5 + 32))
                hourly = day.get("hourly", [{}])
                day_desc = (
                    hourly[len(hourly) // 2].get("weatherDesc", [{}])[0].get("value", "")
                    if hourly
                    else ""
                )
                if use_fahrenheit:
                    forecast.append({
                        "date": date,
                        "max_temp": f"{max_f:.1f}°F",
                        "min_temp": f"{min_f:.1f}°F",
                        "description": day_desc,
                    })
                else:
                    forecast.append({
                        "date": date,
                        "max_temp": f"{max_c:.1f}°C",
                        "min_temp": f"{min_c:.1f}°C",
                        "description": day_desc,
                    })

            result_data = {
                "location": area_name,
                "country": country,
                "temperature": temp_display,
                "feels_like": feels_like_display,
                "humidity": f"{humidity}%",
                "wind_speed": f"{wind_kmph} km/h",
                "wind_direction": wind_dir,
                "visibility": f"{visibility_km} km",
                "description": description,
                "uv_index": uv_index,
                "forecast": forecast,
                "units": units,
            }

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return SkillResult.failure(
                error=f"Unexpected weather data format: {exc}",
                metadata={"location": location},
            )

        return SkillResult.ok(
            data=result_data,
            metadata={"source": "wttr.in", "location_query": location},
        )


__all__ = [
    "WebSearchSkill",
    "URLFetchSkill",
    "WeatherSkill",
]
