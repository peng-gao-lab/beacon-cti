#!/usr/bin/env python3
"""
IoC validator and normalizer
Contains the IoC regex pattern definitions
Supports extracting domains in obfuscated formats such as [.]
"""

import re
from typing import Dict, List


class IOCValidator:
    """IoC validator and normalizer"""
    
    # List of valid top-level domains (TLDs)
    # Includes common generic TLDs, country-code TLDs, and new generic TLDs
    VALID_TLDS = {
        # Generic TLDs
        "com", "org", "net", "edu", "gov", "mil", "int",
        # Country-code TLDs (some common ones)
        "cn", "uk", "jp", "de", "fr", "au", "ca", "in", "ru", "br", "kr", "it", "es", "nl", "se", "no", "pl", "tr", "mx", "za",
        # New generic TLDs (some common ones)
        "io", "co", "ai", "app", "dev", "tech", "online", "site", "cloud",
        "live", "date", "me", "info", "xyz", "top", "win", "bid", "click", "download",
        "stream", "store", "shop", "news", "blog", "email", "space", "website", "host",
        # Other common TLDs
        "tv", "cc", "ws", "name", "mobi", "asia", "tel", "pro", "biz", "us", "eu",
        # Security-relevant TLDs (possibly used by malware)
        "tk", "ml", "ga", "cf", "gq", "pw", "review", "science", "work", "accountant"
    }
    
    # Exhaustive IoC regex patterns
    # Note: domain patterns need to support the [.] obfuscated format
    IOC_PATTERNS = {
        "ip": [
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IPv4
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",  # IPv6
            r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b",  # IPv6 compressed format
        ],
        "domain": [
            # Standard domains (supports plain dots and [.] obfuscation)
            # Matches e.g.: example.com, example[.]com, mol.com-ho[.]me, ntg-sa[.]com, www1.chrome-up[.]date
            # 
            # Key points:
            # 1. Supports [.] as a separator (a common obfuscation technique in malware analysis)
            # 2. Must match the full domain and not stop early
            # 3. Must end with a TLD (at least 2 letters)
            
            # Pattern 1: standard format, supports mixing . and [.]
            # Matches: example.com, example[.]com, ntg-sa[.]com
            # Format: label(.|\[.\])label(.|\[.\])*tld
            r"\b[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.|\[\.\])[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.|\[\.\])*[a-zA-Z]{2,}\b",
            
            # Pattern 2: supports compound formats such as mol.com-ho[.]me, its.com-ho[.]me
            # This pattern specifically handles compound domains containing hyphens
            # Format: label(.|\[.\])label-hyphen-label(.|\[.\])tld
            r"\b[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.|\[\.\])[a-zA-Z0-9\-]+(?:\.|\[\.\])[a-zA-Z]{2,}\b",
            
            # Pattern 3: a more general pattern allowing any number of subdomains and separator combinations
            # Matches: www1.chrome-up[.]date, service.chrome-up[.]date
            r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)(?:\.|\[\.\])+(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)(?:\.|\[\.\])*[a-zA-Z]{2,}\b",
            
            # Pattern 4: domains containing IDNs
            r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)(?:\.|\[\.\])(?:[a-zA-Z]{2,}|xn--[a-zA-Z0-9]+)\b",
        ],
        "email": [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            r"\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Z|a-z]{2,}\b",
            # Email addresses with [.] obfuscation
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+(?:\.|\[\.\])[A-Z|a-z]{2,}\b",
        ],
        "url": [
            # HTTP/HTTPS URL - matches up to a space, quote, or punctuation
            # Note: match the full URL first, then strip trailing quotes etc. in post-processing
            r"https?://[^\s<>\"']+",  # HTTP/HTTPS URL (matches up to a space, quote, or angle bracket)
            r"ftp://[^\s<>\"']+",  # FTP URL (matches up to a space, quote, or angle bracket)
            r"www\.[^\s<>\"']+",  # URL starting with www (matches up to a space, quote, or angle bracket)
            r"[a-zA-Z0-9][a-zA-Z0-9\-]{0,61}[a-zA-Z0-9]?\.[a-zA-Z]{2,}/[^\s<>\"']*",  # URL without a protocol
            # URLs with [.] obfuscation
            r"https?://[^\s<>\"']*\[\.\][^\s<>\"']+",
        ],
        "md5": [
            r"\b[a-fA-F0-9]{32}\b",
        ],
        "sha1": [
            r"\b[a-fA-F0-9]{40}\b",
        ],
        "sha256": [
            r"\b[a-fA-F0-9]{64}\b",
        ],
        "sha512": [
            r"\b[a-fA-F0-9]{128}\b",
        ],
        "cve": [
            r"\bCVE-\d{4}-\d{4,7}\b",
            r"\bcve-\d{4}-\d{4,7}\b",
        ],
        "file_path": [
            # Windows paths: must match one of the following formats
            # 1. Drive paths: C:\path\to\file or C:\temp
            # Requirement: at least one path segment (directory or filename)
            r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n\s]+\\)*[^\\/:*?\"<>|\r\n\s]+",
            # 2. Environment-variable paths: %TEMP%\file.exe or %TEMP%\path\to\file
            # Requirement: the environment variable must be followed by a backslash and at least one path segment
            r"%[A-Za-z0-9_]+%\\(?:[^\\/:*?\"<>|\r\n\s]+\\)*[^\\/:*?\"<>|\r\n\s]+",
            # 3. UNC paths: \\server\share\path or \\server\share
            # Requirement: at least two segments, server and share
            r"\\\\[\\w. -]+\\(?:[^\\/:*?\"<>|\r\n\s]+\\)*[^\\/:*?\"<>|\r\n\s]*",
            # Unix paths: must start with / and contain at least one non-empty segment after a separator
            # Requirement: at least two path segments (e.g. /path/to or /path/file)
            r"/(?:[^/\r\n\s]+/)+[^/\r\n\s]+",
        ],
        # "mutex": [
        #     # GUID-format mutex (e.g. {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX})
        #     r"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}",
        #     # Windows namespace prefixes (Global\ or Local\)
        #     r"\b(?:Global|Local)\\[A-Za-z0-9_\-]{4,}\b",
        #     # Random strings containing digits and letters (at least one digit, at least 6 characters long)
        #     # This avoids matching plain words while still matching mutexes like "mpk1", "AB12CD34"
        #     r"\b(?=[A-Za-z]*[0-9])(?=[0-9]*[A-Za-z])[A-Za-z0-9_\-]{6,}\b",
        #     # Short random strings (4-5 characters, but must mix digits and letters)
        #     r"\b(?=[A-Za-z]*[0-9])(?=[0-9]*[A-Za-z])[A-Za-z0-9_\-]{4,5}\b",
        # ],
        "registry_key": [
            r"HKEY_[A-Z_]+\\[^\s]+",  # Windows registry key
            r"HKLM\\[^\s]+",
            r"HKCU\\[^\s]+",
        ],
    }
    
    @staticmethod
    def validate_ioc(ioc_type: str, value: str) -> bool:
        """
        Validate whether an IoC is valid
        
        Args:
            ioc_type: IoC type
            value: IoC value
        
        Returns:
            True if valid, False otherwise
        """
        if ioc_type == "ip":
            # Validate IPv4
            if re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", value):
                octets = value.split(".")
                try:
                    octets = [int(o) for o in octets]
                    # Exclude private/reserved IPs (but kept, since attackers may use them)
                    # Only validate the format
                    return all(0 <= o <= 255 for o in octets)
                except ValueError:
                    return False
            # IPv6 format validation (simplified)
            return True
        
        elif ioc_type == "domain":
            # Exclude common non-IoC domains
            excluded = ["example.com", "localhost", "test.com", "example.org"]
            # Normalize before checking (convert [.] to .)
            normalized = IOCValidator.normalize_ioc_value(ioc_type, value)
            if normalized.lower() in excluded:
                return False
            
            # Basic length check
            if len(value) <= 3:
                return False
            
            # Extract the top-level domain (TLD) for validation
            # Handle the [.] obfuscated format
            domain_parts = normalized.split(".")
            if len(domain_parts) < 2:
                return False
            
            tld = domain_parts[-1].lower()
            
            # Check that the TLD is in the valid list
            # This filters out false positives like MagicHound.Leash
            if tld not in IOCValidator.VALID_TLDS:
                return False
            
            return True
        
        elif ioc_type == "email":
            return "@" in value and ("." in value.split("@")[1] or "[.]" in value.split("@")[1])
        
        elif ioc_type == "md5":
            return len(value) == 32 and all(c in "0123456789abcdefABCDEF" for c in value)
        
        elif ioc_type == "sha1":
            return len(value) == 40 and all(c in "0123456789abcdefABCDEF" for c in value)
        
        elif ioc_type == "sha256":
            return len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)
        
        elif ioc_type == "sha512":
            return len(value) == 128 and all(c in "0123456789abcdefABCDEF" for c in value)
        
        elif ioc_type == "file_path":
            # A file path should not contain newline characters
            if '\n' in value or '\r' in value:
                return False
            
            # Strip quotes (may have been included during extraction)
            clean_value = value.strip('"\'')
            
            # 1. Short-token stoplist: a /x whose length is <= 3 and all letters/digits is dropped directly
            # E.g.: /c, /1.1, /port, /free
            if clean_value.startswith('/') or clean_value.startswith('\\'):
                # Extract the first path segment
                first_segment = clean_value.lstrip('/\\').split('/')[0].split('\\')[0]
                # If the first segment has length <= 3 and contains only letters or digits, it is most likely a false positive
                if len(first_segment) <= 3 and first_segment.replace('.', '').isalnum():
                    # Exclude obvious file extensions (e.g. .exe, .dll)
                    if not (len(first_segment) > 1 and first_segment.startswith('.')):
                        return False
            
            # 2. Exclude URL paths: paths containing a domain or IP address
            # Check whether it contains a domain pattern (e.g. github.com, 139.59.46.154)
            domain_pattern = r'[a-zA-Z0-9\-]+(?:\.|\[\.\])[a-zA-Z]{2,}'
            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            
            # If the path contains a domain or IP, it is most likely a URL path
            if re.search(domain_pattern, clean_value) or re.search(ip_pattern, clean_value):
                # Check whether it is a file extension (at the end of the path)
                # File extensions are usually 2-4 characters and not in the middle of the path
                if not (clean_value.endswith(('.exe', '.dll', '.sys', '.bat', '.cmd', '.ps1', '.vbs', '.js', '.tmp', '.log', '.txt', '.conf', '.cfg', '.ini'))):
                    return False
            
            # 3. Exclude registry path fragments
            # E.g.: \\Microsoft, \\CurrentVersion, \\temp
            # These usually start with \\ but have only one segment, and may be part of a registry key
            if clean_value.startswith('\\\\'):
                segments = clean_value[2:].split('\\')
                # If there is only one segment and it looks like a registry key name (capitalized, common registry key name)
                if len(segments) == 1 or (len(segments) == 2 and segments[1] == ''):
                    segment = segments[0]
                    # Common registry key name pattern (capitalized, CamelCase)
                    if segment and segment[0].isupper() and len(segment) > 3:
                        # Check whether it is a common registry key name
                        common_registry_keys = ['Microsoft', 'CurrentVersion', 'Software', 'System', 'Classes', 'Services']
                        if segment in common_registry_keys:
                            return False
            
            # 4. Validate the path structure: must contain at least one non-empty segment after a separator
            # Windows paths: must contain something after at least one backslash
            if ':' in clean_value and '\\' in clean_value:
                # C:\ or C:\file format
                parts = clean_value.split('\\', 1)
                if len(parts) < 2 or not parts[1]:
                    return False
            elif clean_value.startswith('%'):
                # %TEMP% or %TEMP%\file format
                if not '\\' in clean_value:
                    # Only an environment variable name with no path; reject
                    return False
                parts = clean_value.split('\\', 1)
                if len(parts) < 2 or not parts[1]:
                    return False
            elif clean_value.startswith('\\\\'):
                # UNC paths: \\server\share\path or \\server\share
                parts = clean_value[2:].split('\\')
                # At least two non-empty segments are required (server + share)
                non_empty_parts = [p for p in parts if p]
                if len(non_empty_parts) < 2:
                    # If there is only one segment, it is most likely a registry path fragment (e.g. \\temp, \\Microsoft)
                        return False
            elif clean_value.startswith('/'):
                # Unix paths: /path/to/file
                # Per the regex, at least the /path/to/file format is required (at least two separators)
                parts = clean_value.split('/')
                non_empty_parts = [p for p in parts if p]
                # At least two non-empty segments are required (e.g. /path/to or /path/file)
                if len(non_empty_parts) < 2:
                        return False
            
            # 5. Exclude things that look like an HTTP protocol version (e.g. /1.1)
            if re.match(r'^/\d+\.\d+$', clean_value):
                        return False
            
            return True
        
        elif ioc_type == "mutex":
            # A mutex is usually a short mixed alphanumeric string (e.g. mpk1, AB12CD34)
            # Length is usually between 4 and 32 characters
            clean_value = value.strip()
            if len(clean_value) < 4 or len(clean_value) > 32:
                return False
            # Should contain letters and digits (at least one of each)
            has_letter = bool(re.search(r'[a-zA-Z]', clean_value))
            has_digit = bool(re.search(r'[0-9]', clean_value))
            # Underscores and hyphens are allowed
            if not (has_letter and has_digit):
                # If it is only letters or only digits, it may be a false positive
                return False
            # Only letters, digits, underscores, and hyphens are allowed
            if not re.match(r'^[a-zA-Z0-9_\-]+$', clean_value):
                return False
            return True
        
        elif ioc_type == "aes_key":
            # An AES key is usually a longer alphanumeric string (e.g. agkrhfpdbvhdhrkj)
            # Length is usually between 16 and 64 characters
            clean_value = value.strip()
            if len(clean_value) < 16 or len(clean_value) > 64:
                return False
            # Usually contains only letters and digits
            if not re.match(r'^[a-zA-Z0-9]+$', clean_value):
                return False
            return True
        
        # Default: accept the IoC
        return True
    
    @staticmethod
    def normalize_ioc_value(ioc_type: str, value: str) -> str:
        """
        Normalize an IoC value for deduplication
        Unifies the various string forms:
        - JSON escaping: unify the number of backslashes (c:\\\\temp -> c:\\temp)
        - Defanging symbols: unify [.] to . (139.59.46[.]154 -> 139.59.46.154)
        - URL spaces: remove spaces
        - hxxp:// is unified to http://
        
        Args:
            ioc_type: IoC type
            value: IoC value
        
        Returns:
            The normalized value
        """
        # First strip leading/trailing whitespace and quotes
        normalized = value.strip().strip('"\'')
        
        # URLs need special handling
        if ioc_type == "url":
            # Unify hxxp:// to http://
            normalized = re.sub(r'^hxxp://', 'http://', normalized, flags=re.IGNORECASE)
            # Remove spaces from the URL
            normalized = normalized.replace(' ', '')
            # Replace [.] with .
            normalized = normalized.replace("[.]", ".")
            return normalized.lower()
        
        # For domains and emails, convert [.] to .
        elif ioc_type in ["domain", "email"]:
            normalized = normalized.replace("[.]", ".")
            return normalized.lower()
        
        # For IP addresses, convert [.] to .
        elif ioc_type == "ip":
            normalized = normalized.replace("[.]", ".")
            return normalized.lower()
        
        # For file paths and registry keys, unify the number of backslashes
        elif ioc_type in ["file_path", "registry_key"]:
            # Collapse multiple consecutive backslashes into one (but keep the leading \\ of UNC paths)
            if normalized.startswith('\\\\'):
                # UNC path: keep the leading \\ and unify the remaining backslashes
                normalized = '\\\\' + re.sub(r'\\+', r'\\', normalized[2:])
            else:
                # Regular path: unify all backslashes
                normalized = re.sub(r'\\+', r'\\', normalized)
            return normalized.lower()
        
        # For hashes, use lowercase
        elif ioc_type in ["md5", "sha1", "sha256", "sha512"]:
            return normalized.lower()
        
        # For CVEs, use uppercase
        elif ioc_type == "cve":
            return normalized.upper()
        
        # For other types such as mutex and aes_key, use lowercase and remove spaces
        else:
            normalized = normalized.replace(' ', '')
            return normalized.lower()
    
    @classmethod
    def get_compiled_patterns(cls) -> Dict[str, List[re.Pattern]]:
        """
        Get the compiled regex patterns
        
        Returns:
            Dict whose keys are IoC types and values are lists of compiled regexes
        """
        compiled_patterns = {}
        for ioc_type, patterns in cls.IOC_PATTERNS.items():
            compiled_patterns[ioc_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        return compiled_patterns

