from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit


SUPPORTED_IOC_TYPES = {
    "ip",
    "domain",
    "email",
    "url",
    "md5",
    "sha1",
    "sha256",
    "sha512",
    "cve",
    "file_path",
    "registry_key",
}

IOC_TYPE_ALIASES = {
    "filepath": "file_path",
    "registrykey": "registry_key",
}

_WRAP_CHARS = set("()[]{}<>\"'“”‘’,;:")
_TERMINAL_CHARS = set(".,;:)]}>\"'")
_ORG_SUFFIXES = {"inc", "ltd", "llc", "corp", "co", "gmbh", "plc", "sa", "s.a."}
_PLATFORM_TOKENS = {"x86", "x64", "win32", "windows", "linux"}


def strip_wrappers(raw: str) -> str:
    text = (raw or "").strip()
    changed = True
    while text and changed:
        changed = False
        while text and text[0] in _WRAP_CHARS:
            text = text[1:].lstrip()
            changed = True
        while text and text[-1] in _WRAP_CHARS:
            text = text[:-1].rstrip()
            changed = True
    return text


def strip_terminal_punctuation(raw: str) -> str:
    text = (raw or "").rstrip()
    while text and text[-1] in _TERMINAL_CHARS:
        text = text[:-1].rstrip()
    return text


def _defang_dots(text: str) -> str:
    return re.sub(r"\[\.\]", ".", text, flags=re.IGNORECASE)


def _strip_outer_quotes(text: str) -> str:
    while len(text) >= 2 and (
        (text[0] == text[-1] and text[0] in {"'", '"'})
        or (text[0], text[-1]) in {("“", "”"), ("‘", "’")}
    ):
        text = text[1:-1].strip()
    return text


def canonicalize_ip(raw: str) -> str:
    text = strip_terminal_punctuation(_defang_dots(strip_wrappers(raw)))
    parts = text.split(".")
    if len(parts) == 4 and all(re.fullmatch(r"\d{1,3}", p or "") for p in parts):
        nums = [int(p) for p in parts]
        if all(0 <= n <= 255 for n in nums):
            return ".".join(str(n) for n in nums)
    try:
        ip_obj = ipaddress.ip_address(text)
    except ValueError:
        return text.lower()
    return ip_obj.compressed.lower()


def canonicalize_domain(raw: str) -> str:
    text = _defang_dots(strip_wrappers(raw))
    text = strip_terminal_punctuation(text).lower()
    return text.rstrip(".")


def canonicalize_email(raw: str) -> str:
    text = strip_terminal_punctuation(strip_wrappers(raw))
    if "@" not in text:
        return text.lower()
    local, domain = text.split("@", 1)
    return f"{local.lower()}@{canonicalize_domain(domain)}"


def canonicalize_url(raw: str) -> str:
    text = strip_wrappers(raw).strip()
    text = re.sub(r"^hxxps://", "https://", text, flags=re.IGNORECASE)
    text = re.sub(r"^hxxp://", "http://", text, flags=re.IGNORECASE)
    text = text.replace("[:]//", "://")
    text = _defang_dots(text)
    text = strip_terminal_punctuation(text)
    if not text:
        return text

    has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text))
    parse_target = text
    if not has_scheme and (
        text.lower().startswith("www.")
        or bool(re.search(r"[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", text))
    ):
        parse_target = f"http://{text}"

    try:
        split = urlsplit(parse_target)
    except ValueError:
        return strip_terminal_punctuation(parse_target.lower())

    scheme = split.scheme.lower()
    host = (split.hostname or "").lower()
    try:
        port = split.port
    except ValueError:
        port = None

    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None

    userinfo = ""
    if split.username:
        userinfo = split.username
        if split.password:
            userinfo += f":{split.password}"
        userinfo += "@"

    if ":" in host and not host.startswith("["):
        host_display = f"[{host}]"
    else:
        host_display = host
    if port is not None:
        host_display = f"{host_display}:{port}"
    netloc = f"{userinfo}{host_display}" if host_display else split.netloc.lower()

    canonical = urlunsplit((scheme, netloc, split.path, split.query, split.fragment))
    return strip_terminal_punctuation(canonical)


def canonicalize_hash(raw: str) -> str:
    text = strip_terminal_punctuation(strip_wrappers(raw)).lower()
    if text.startswith("0x"):
        text = text[2:]
    return text


def canonicalize_cve(raw: str) -> str:
    text = strip_terminal_punctuation(strip_wrappers(raw))
    text = text.replace("–", "-").replace("—", "-")
    text = text.upper()
    match = re.search(r"CVE[-_ ]?(\d{4})[-_ ]?(\d{4,7})", text)
    if match:
        return f"CVE-{match.group(1)}-{match.group(2)}"
    return text


def canonicalize_file_path(raw: str) -> str:
    text = _strip_outer_quotes(strip_wrappers(raw))
    text = re.sub(r"\[\.\](?=\.)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\.)\[\.\]", "", text, flags=re.IGNORECASE)
    text = _defang_dots(text)
    text = re.sub(r"%([A-Za-z0-9_]+)%", lambda m: f"%{m.group(1).upper()}%", text)
    if not text:
        return text

    is_windows_like = (
        bool(re.match(r"^[A-Za-z]:[\\/]", text))
        or text.startswith("\\\\")
        or ("\\" in text and not text.startswith("/"))
    )

    if is_windows_like:
        text = text.replace("/", "\\")
        if re.match(r"^[A-Za-z]:", text):
            text = f"{text[0].upper()}{text[1:]}"
        if text.startswith("\\"):
            text = "\\\\" + text.lstrip("\\")
            unc_match = re.match(r"^\\\\([^\\]+)\\([^\\]+)(.*)$", text)
            if unc_match:
                text = (
                    "\\\\"
                    + unc_match.group(1).lower()
                    + "\\"
                    + unc_match.group(2).lower()
                    + unc_match.group(3)
                )
        if text.endswith("\\"):
            if not re.fullmatch(r"[A-Z]:\\", text):
                if not re.fullmatch(r"\\\\[^\\]+\\[^\\]+\\?", text):
                    text = text.rstrip("\\")
    else:
        if text != "/":
            text = text.rstrip("/")
    return text


_REGISTRY_HIVE_MAP = {
    "HKEY_LOCAL_MACHINE": "HKLM",
    "HKLM": "HKLM",
    "HKEY_CURRENT_USER": "HKCU",
    "HKCU": "HKCU",
    "HKEY_CLASSES_ROOT": "HKCR",
    "HKCR": "HKCR",
    "HKEY_USERS": "HKU",
    "HKU": "HKU",
    "HKEY_CURRENT_CONFIG": "HKCC",
    "HKCC": "HKCC",
}


def canonicalize_registry_key(raw: str) -> str:
    text = _strip_outer_quotes(strip_wrappers(raw))
    text = text.replace("/", "\\")
    text = re.sub(r"\\+", r"\\", text)
    if not text:
        return text
    parts = text.split("\\", 1)
    hive = parts[0].upper()
    mapped = _REGISTRY_HIVE_MAP.get(hive, hive)
    if len(parts) == 1:
        return mapped
    tail = parts[1]
    canonical = f"{mapped}\\{tail}"
    if canonical.endswith("\\") and "\\" in canonical:
        canonical = canonical.rstrip("\\")
    return canonical


def normalize_ioc_type(ioc_type: str) -> str:
    normalized = (ioc_type or "").strip().lower()
    return IOC_TYPE_ALIASES.get(normalized, normalized)


def canonicalize_ioc_value(ioc_type: str, raw_value: str) -> tuple[str, str]:
    ioc_type_l = normalize_ioc_type(ioc_type)
    raw = raw_value if raw_value is not None else ""
    if ioc_type_l == "ip":
        return raw, canonicalize_ip(raw)
    if ioc_type_l == "domain":
        return raw, canonicalize_domain(raw)
    if ioc_type_l == "email":
        return raw, canonicalize_email(raw)
    if ioc_type_l == "url":
        return raw, canonicalize_url(raw)
    if ioc_type_l in {"md5", "sha1", "sha256", "sha512"}:
        return raw, canonicalize_hash(raw)
    if ioc_type_l == "cve":
        return raw, canonicalize_cve(raw)
    if ioc_type_l == "file_path":
        return raw, canonicalize_file_path(raw)
    if ioc_type_l == "registry_key":
        return raw, canonicalize_registry_key(raw)
    return raw, strip_terminal_punctuation(strip_wrappers(raw))


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_basic(text: str, *, replace_separators: bool = True) -> str:
    out = (text or "").lower().strip()
    if replace_separators:
        out = re.sub(r"[-_/]+", " ", out)
    return collapse_spaces(out)


def normalize_joined(text: str) -> str:
    return normalize_basic(text).replace(" ", "")


def extract_actor_canonical_id(text: str) -> str | None:
    match = re.search(r"\b(APT|UNC|TA|FIN)\s*[-_ ]?\s*0*([0-9]{1,6})\b", text or "", re.I)
    if not match:
        return None
    return f"{match.group(1).upper()}{int(match.group(2))}"


def extract_digit_sequence(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", text or ""))


def strip_campaign_prefix_once(text: str) -> str:
    return re.sub(r"^(operation|campaign)\s+", "", text or "", count=1)


def normalize_us_abbr(text: str) -> str:
    out = text or ""
    out = re.sub(r"\bu\s*\.?\s*s\.?\b", "us", out, flags=re.IGNORECASE)
    return collapse_spaces(out.lower())


def strip_org_suffix_once(text: str) -> str:
    normalized = collapse_spaces(text or "")
    if not normalized:
        return normalized
    parts = normalized.split(" ")
    if parts and parts[-1].lower() in _ORG_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts).strip()


def strip_tail_version_platform(text: str) -> str:
    normalized = collapse_spaces(text or "")
    normalized = re.sub(
        r"\((x86|x64|win32|windows|linux)\)\s*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    parts = normalized.split(" ")
    if not parts:
        return normalized
    if parts[-1].lower() in _PLATFORM_TOKENS:
        parts = parts[:-1]
    if parts and re.fullmatch(r"v?\d+(?:\.\d+)*", parts[-1], flags=re.IGNORECASE):
        parts = parts[:-1]
    return collapse_spaces(" ".join(parts))
