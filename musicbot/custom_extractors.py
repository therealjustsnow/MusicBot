"""
Custom extractor support for MusicBot.

Users can drop .properties files into config/extractors/ to define external
scripts that MusicBot will call when a URL matches the extractor's regex.
The working directory when running a script is the extractors folder itself.

Extractor definition format (config/extractors/my-extractor.properties):

    windows_command = python myscript.py --url {url} --cache {audio_cache}
    unix_command    = python3 myscript.py --url {url} --cache {audio_cache}
    url_regex       = https?://example\\.com/.*

Variables substituted in commands:
    {url}          -- the input URL being extracted
    {audio_cache}  -- absolute path to the bot's audio cache directory

The script should write JSON to stdout.  Two response types are supported:

Track response:
    {
        "track": {
            "name":      "Track Title",
            "thumbnail": "https://...",
            "result":    "https://direct-audio-url.mp3"
        }
    }
    The "result" field is required.  It may also be "audio-cache://filename"
    when the script itself has already placed the file in the audio cache.

Album response:
    {
        "album": {
            "name":   "Album Name",
            "result": ["https://track1", "https://track2", ...]
        }
    }
    The "result" field (list of external URLs) is required.

An empty JSON object {} is treated as "no result"; the next extractor or the
default yt-dlp logic will be tried instead.
"""

import asyncio
import configparser
import json
import logging
import platform
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .bot import MusicBot

log = logging.getLogger(__name__)


@dataclass
class CustomExtractorDefinition:
    """A single custom extractor loaded from a .properties file."""

    name: str
    source_file: Path
    windows_command: str
    unix_command: str
    url_regex: "re.Pattern[str]"

    @property
    def extractor_name(self) -> str:
        return f"custom:{self.name}"

    def get_command(self) -> str:
        if platform.system() == "Windows":
            return self.windows_command
        return self.unix_command


class CustomExtractorManager:
    """Manages loading and running custom extractors."""

    def __init__(self, bot: "MusicBot") -> None:
        self.bot = bot
        self._extractors: List[CustomExtractorDefinition] = []
        # Cache maps "extractor_name:url" -> parsed response dict or Exception
        self._response_cache: Dict[str, Any] = {}

    @property
    def extractors_path(self) -> Path:
        return self.bot.config.custom_extractors_path

    def load_extractors(self) -> int:
        """
        Scan the extractors directory and load all .properties files.
        Clears the response cache.  Returns the number of extractors loaded.
        """
        self._extractors = []
        self._response_cache = {}

        if not self.extractors_path.is_dir():
            log.debug(
                "Custom extractors directory does not exist: %s", self.extractors_path
            )
            return 0

        count = 0
        for fpath in sorted(self.extractors_path.glob("*.properties")):
            try:
                extractor = self._load_definition(fpath)
                if extractor:
                    self._extractors.append(extractor)
                    count += 1
                    log.info("Loaded custom extractor: %s", extractor.extractor_name)
            except Exception:  # pylint: disable=broad-exception-caught
                log.exception("Failed to load custom extractor from: %s", fpath)

        log.info("Loaded %d custom extractor(s).", count)
        return count

    def _load_definition(self, fpath: Path) -> Optional[CustomExtractorDefinition]:
        """Parse a .properties file into a CustomExtractorDefinition."""
        parser = configparser.ConfigParser()
        with open(fpath, encoding="utf-8") as fh:
            # configparser requires at least one section header
            parser.read_string("[Extractor]\n" + fh.read())

        section = "Extractor"
        windows_cmd = parser.get(section, "windows_command", fallback="").strip()
        unix_cmd = parser.get(section, "unix_command", fallback="").strip()
        url_regex_str = parser.get(section, "url_regex", fallback="").strip()

        if not windows_cmd and not unix_cmd:
            log.error(
                "Custom extractor '%s' missing both windows_command and unix_command.",
                fpath.name,
            )
            return None

        if not url_regex_str:
            log.error("Custom extractor '%s' missing url_regex.", fpath.name)
            return None

        try:
            url_regex = re.compile(url_regex_str)
        except re.error as e:
            log.error("Custom extractor '%s' has invalid url_regex: %s", fpath.name, e)
            return None

        return CustomExtractorDefinition(
            name=fpath.stem,
            source_file=fpath,
            windows_command=windows_cmd,
            unix_command=unix_cmd,
            url_regex=url_regex,
        )

    def match_url(self, url: str) -> Optional[CustomExtractorDefinition]:
        """Return the first extractor whose regex matches url, or None."""
        for extractor in self._extractors:
            if extractor.url_regex.search(url):
                return extractor
        return None

    def clear_cache(self) -> None:
        """Discard all cached extractor responses."""
        self._response_cache = {}

    async def run_extractor(
        self,
        extractor: CustomExtractorDefinition,
        url: str,
    ) -> Dict[str, Any]:
        """
        Execute the custom extractor script for the given URL and return the
        parsed JSON response.  The result is cached so the script is not
        re-run for the same URL.

        Returns an empty dict if the script outputs nothing or "{}".

        Raises ValueError on script errors or invalid output.
        """
        cache_key = f"{extractor.name}:{url}"
        if cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            if isinstance(cached, Exception):
                raise cached
            return dict(cached)

        command_template = extractor.get_command()
        if not command_template:
            err = ValueError(
                f"No command configured for platform '{platform.system()}' "
                f"in extractor '{extractor.name}'"
            )
            self._response_cache[cache_key] = err
            raise err

        audio_cache = str(self.bot.config.audio_cache_path.resolve())
        command_str = command_template.replace("{url}", url).replace(
            "{audio_cache}", audio_cache
        )

        working_dir = str(extractor.source_file.parent.resolve())

        try:
            if platform.system() == "Windows":
                proc = await asyncio.create_subprocess_shell(
                    command_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *shlex.split(command_str),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                )
            stdout, stderr = await proc.communicate()
        except (OSError, FileNotFoundError) as e:
            err = ValueError(
                f"Failed to launch extractor '{extractor.name}': {e}"
            )
            self._response_cache[cache_key] = err
            raise err

        if stderr:
            log.debug(
                "Custom extractor '%s' stderr:\n%s",
                extractor.name,
                stderr.decode("utf-8", errors="replace"),
            )

        if proc.returncode != 0:
            err = ValueError(
                f"Custom extractor '{extractor.name}' exited with code "
                f"{proc.returncode}: "
                + stderr.decode("utf-8", errors="replace")[:400]
            )
            self._response_cache[cache_key] = err
            raise err

        stdout_text = stdout.decode("utf-8", errors="replace").strip()

        if not stdout_text or stdout_text == "{}":
            self._response_cache[cache_key] = {}
            return {}

        try:
            result: Dict[str, Any] = json.loads(stdout_text)
        except json.JSONDecodeError as e:
            err = ValueError(
                f"Custom extractor '{extractor.name}' returned invalid JSON: {e}\n"
                f"Output: {stdout_text[:300]}"
            )
            self._response_cache[cache_key] = err
            raise err

        if not isinstance(result, dict):
            err = ValueError(
                f"Custom extractor '{extractor.name}' must return a JSON object"
            )
            self._response_cache[cache_key] = err
            raise err

        has_track = "track" in result
        has_album = "album" in result

        if has_track and has_album:
            err = ValueError(
                f"Custom extractor '{extractor.name}' returned both 'track' and "
                f"'album' — these are mutually exclusive"
            )
            self._response_cache[cache_key] = err
            raise err

        if has_track:
            track = result["track"]
            if not isinstance(track, dict) or "result" not in track:
                err = ValueError(
                    f"Custom extractor '{extractor.name}' track response is missing "
                    f"the required 'result' field"
                )
                self._response_cache[cache_key] = err
                raise err

        if has_album:
            album = result["album"]
            if not isinstance(album, dict) or "result" not in album:
                err = ValueError(
                    f"Custom extractor '{extractor.name}' album response is missing "
                    f"the required 'result' field"
                )
                self._response_cache[cache_key] = err
                raise err
            if not isinstance(album["result"], list):
                err = ValueError(
                    f"Custom extractor '{extractor.name}' album 'result' must be a "
                    f"list of URL strings"
                )
                self._response_cache[cache_key] = err
                raise err

        self._response_cache[cache_key] = result
        return result

    def build_album_data(
        self,
        extractor: CustomExtractorDefinition,
        input_url: str,
        album_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a YtdlpResponseDict-compatible dict for an album response."""
        name: str = album_data.get("name", "") or "album"
        track_urls: List[str] = album_data["result"]

        entries = [
            {
                "__input_subject": track_url,
                "_type": "url",
                "url": track_url,
                "extractor": extractor.extractor_name,
            }
            for track_url in track_urls
        ]

        return {
            "__input_subject": input_url,
            "__header_data": None,
            "_type": "playlist",
            "extractor": extractor.extractor_name,
            "extractor_key": f"Custom{extractor.name.title().replace('-', '').replace('_', '')}",
            "title": name,
            "entries": entries,
            "playlist_count": len(entries),
        }

    def build_audio_cache_track_data(
        self,
        extractor: CustomExtractorDefinition,
        input_url: str,
        track_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build data for a track whose result is an audio-cache:// path."""
        result_url: str = track_data["result"]
        name: str = track_data.get("name", "") or ""
        thumbnail: str = track_data.get("thumbnail", "") or ""

        cache_file = result_url[len("audio-cache://"):]
        cache_path = self.bot.config.audio_cache_path / cache_file
        if not name:
            name = Path(cache_file).stem

        return {
            "__input_subject": input_url,
            "__header_data": None,
            "__expected_filename": str(cache_path.resolve()),
            "_type": "local",
            "extractor": extractor.extractor_name,
            "extractor_key": f"Custom{extractor.name.title().replace('-', '').replace('_', '')}",
            "title": name,
            "url": input_url,
            "webpage_url": input_url,
            "thumbnail": thumbnail,
        }
