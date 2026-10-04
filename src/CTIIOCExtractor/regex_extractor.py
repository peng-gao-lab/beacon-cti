#!/usr/bin/env python3
"""
Regex IoC extractor
"""

import re
from typing import List, Dict, Any
from collections import defaultdict

from .ioc_validator import IOCValidator


class RegexExtractor:
    """Regex IoC extractor"""
    
    def __init__(self):
        """Initialize the extractor"""
        self.validator = IOCValidator()
        self.compiled_patterns = IOCValidator.get_compiled_patterns()
    
    def extract(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Extract IoCs via regex extraction
        
        Args:
            sentences: list of sentences
        
        Returns:
            List of IoCs; each IoC contains type, value, normalized_value, sentence_indices, source
        """
        # Use a dict to aggregate the multiple occurrences of the same IoC
        ioc_dict = defaultdict(lambda: {
            "ioc_type": "",
            "value": "",
            "normalized_value": "",
            "sentence_indices": set()
        })
        
        for ioc_type, compiled_patterns in self.compiled_patterns.items():
            for sentence_idx, sentence in enumerate(sentences):
                for pattern_idx, pattern in enumerate(compiled_patterns):
                    matches = pattern.finditer(sentence)
                    for match in matches:
                        value = match.group(0)
                        
                        # Domains need special handling to make sure the full domain is extracted
                        # Handles cases like mol.com-ho[.]me
                        if ioc_type == "domain":
                            match_end = match.end()
                            remaining_text = sentence[match_end:]
                            
                            # Check whether there are more domain parts to match
                            # E.g. if "mol.com" was matched but "-ho[.]me" follows,
                            # we need to keep matching until a word boundary or a non-domain character
                            
                            # If the current match ends with a separator, look for the following part
                            if value.endswith(".") or value.endswith("[.]"):
                                # Look for the following domain part
                                # Pattern: hyphen + alphanumerics + separator + TLD
                                extended_match = re.search(
                                    r'[a-zA-Z0-9\-]+(?:\.|\[\.\])[a-zA-Z]{2,}',
                                    remaining_text
                                )
                                if extended_match and extended_match.start() == 0:
                                    # Check that the extended part immediately follows the current match
                                    value = value + extended_match.group(0)
                            # If the current match does not end with a separator, check whether it needs extending
                            # E.g. "mol.com" was matched but it should actually be "mol.com-ho[.]me"
                            elif remaining_text and re.match(r'^-[a-zA-Z0-9\-]+(?:\.|\[\.\])', remaining_text):
                                # Look for the following part starting with a hyphen
                                extended_match = re.search(
                                    r'-[a-zA-Z0-9\-]+(?:\.|\[\.\])[a-zA-Z]{2,}',
                                    remaining_text
                                )
                                if extended_match and extended_match.start() == 0:
                                    value = value + extended_match.group(0)
                        
                        # Strip trailing punctuation and newline characters
                        # These punctuation marks may be part of the text formatting and should not be included in the IoC
                        # Note: for URLs, slashes in the path should be kept; only strip trailing punctuation
                        if ioc_type == "url":
                            # A URL may end with a slash, which must be kept; only strip quotes etc.
                            value = value.rstrip('"\'.,;:!?)\\]}>')
                            # Truncate at the first newline (a URL should not span lines)
                            if '\n' in value or '\r' in value:
                                value = value.split('\n')[0].split('\r')[0]
                        elif ioc_type == "file_path":
                            # A file path should not span lines or sentences
                            # Truncate at the first newline, space, or end of sentence
                            if '\n' in value:
                                value = value.split('\n')[0]
                            if '\r' in value:
                                value = value.split('\r')[0]
                            # Strip trailing punctuation
                            value = value.rstrip('"\'.,;:!?)\\]}>')
                        else:
                            # Other types: strip all trailing punctuation
                            value = value.rstrip('"\'.,;:!?)\\]}>/')
                            # Truncate at the first newline (an IoC should not span lines)
                            if '\n' in value or '\r' in value:
                                value = value.split('\n')[0].split('\r')[0]
                        
                        # Basic validation
                        if self.validator.validate_ioc(ioc_type, value):
                            normalized_value = self.validator.normalize_ioc_value(ioc_type, value)
                            key = (ioc_type, normalized_value)
                            
                            if not ioc_dict[key]["ioc_type"]:
                                ioc_dict[key]["ioc_type"] = ioc_type
                                ioc_dict[key]["value"] = value
                                ioc_dict[key]["normalized_value"] = normalized_value
                            
                            ioc_dict[key]["sentence_indices"].add(sentence_idx)
        
        # Convert to list format
        iocs = []
        for key, data in ioc_dict.items():
            iocs.append({
                "ioc_type": data["ioc_type"],
                "value": data["value"],
                "normalized_value": data["normalized_value"],
                "sentence_indices": sorted(list(data["sentence_indices"])),
                "source": "regex"
            })
        
        # Filter out IoCs that are substrings of other IoCs
        # 1. Domains: filtered out if it is a substring of another domain and appears in the same sentence
        # 2. URLs: filtered out if it is a substring of another URL and appears in the same sentence
        # 3. Domains: filtered out if it is part of a URL (e.g. dyndns.org/ is part of http://checkip.dyndns.org/)
        # 4. Mutexes: filtered out if it is part of a URL or domain (commented out; mutex extraction is disabled)
        filtered_iocs = []
        for ioc in iocs:
            is_substring = False
            ioc_value = ioc["value"]
            ioc_type = ioc["ioc_type"]
            
            for other_ioc in iocs:
                if other_ioc["value"] == ioc_value:
                    continue
                
                other_value = other_ioc["value"]
                other_type = other_ioc["ioc_type"]
                
                # Check whether they appear in the same sentence
                if not (set(ioc["sentence_indices"]) & set(other_ioc["sentence_indices"])):
                    continue
                
                # Case 1: same-type IoCs; the shorter one is filtered out if it is part of the longer one
                if ioc_type == other_type and len(other_value) > len(ioc_value):
                    if ioc_value in other_value:
                        is_substring = True
                        break
                
                # Case 2: the domain is part of a URL (e.g. dyndns.org/ is part of http://checkip.dyndns.org/)
                if ioc_type == "domain" and other_type == "url":
                    # Check whether the domain (possibly with a slash) is part of the URL
                    domain_with_slash = ioc_value + "/"
                    if ioc_value in other_value or domain_with_slash in other_value:
                        is_substring = True
                        break
                
                # # Case 3: the mutex is part of a URL (e.g. rapid7 in https://github.com/rapid7/..., 0w0O6 in a URL path)
                # if ioc_type == "mutex" and other_type == "url":
                #     # Check whether the mutex value is in the URL
                #     # Must make sure it is part of the path or domain, not standalone
                #     if ioc_value in other_value:
                #         # Check the context: if it is preceded/followed by / or . or :, it is most likely part of the URL
                #         idx = other_value.find(ioc_value)
                #         if idx >= 0:
                #             before = other_value[max(0, idx-1):idx]
                #             after = other_value[idx+len(ioc_value):idx+len(ioc_value)+1]
                #             # If preceded/followed by a URL separator (/, :, .), it is most likely part of the URL
                #             if before in ['/', '.', ':'] or after in ['/', '.', ':']:
                #                 is_substring = True
                #                 break
                # 
                # # Case 4: the mutex is part of a domain (e.g. rapid7 in rapid7.github.com)
                # if ioc_type == "mutex" and other_type == "domain":
                #     # Check whether the mutex value is in the domain
                #     if ioc_value in other_value:
                #         idx = other_value.find(ioc_value)
                #         if idx >= 0:
                #             before = other_value[max(0, idx-1):idx]
                #             after = other_value[idx+len(ioc_value):idx+len(ioc_value)+1]
                #             # If preceded/followed by a domain separator (.), it is most likely part of the domain
                #             if before == '.' or after == '.':
                #                 is_substring = True
                #                 break
                # 
                # # Case 5: check whether the mutex is in a URL context within the sentence (even if no URL IoC was extracted)
                # # E.g. the sentence contains "https://github.com/rapid7/..." or "hxxp://..." but URL extraction may have failed
                # if ioc_type == "mutex" and not is_substring:
                #     for sentence_idx in ioc["sentence_indices"]:
                #         sentence = sentences[sentence_idx]
                #         # Check whether the mutex value is in the context of a URL pattern
                #         # Look for URL patterns containing the mutex value (including obfuscated forms such as hxxp://)
                #         url_patterns = [
                #             r'https?://[^\s<>"\']*' + re.escape(ioc_value) + r'[^\s<>"\']*',
                #             r'hxxp://[^\s<>"\']*' + re.escape(ioc_value) + r'[^\s<>"\']*',  # supports the hxxp obfuscated format
                #             r'ftp://[^\s<>"\']*' + re.escape(ioc_value) + r'[^\s<>"\']*',
                #             r'[a-zA-Z0-9][a-zA-Z0-9\-]*\.[a-zA-Z]{2,}/[^\s<>"\']*' + re.escape(ioc_value) + r'[^\s<>"\']*',
                #             # Also check the URL path part (e.g. :4443/0w0O6)
                #             r'://[^\s<>"\']*' + re.escape(ioc_value) + r'[^\s<>"\']*',
                #         ]
                #         for pattern in url_patterns:
                #             match = re.search(pattern, sentence, re.IGNORECASE)
                #             if match:
                #                 url_text = match.group(0)
                #                 idx = url_text.find(ioc_value)
                #                 if idx >= 0:
                #                     before = url_text[max(0, idx-1):idx]
                #                     after = url_text[idx+len(ioc_value):idx+len(ioc_value)+1]
                #                     # If preceded/followed by a URL separator (/, :, .), it is most likely part of the URL
                #                     if before in ['/', '.', ':'] or after in ['/', '.', ':', '"', "'"]:
                #                         is_substring = True
                #                         break
                #         if is_substring:
                #             break
            
            if not is_substring:
                filtered_iocs.append(ioc)
        
        return filtered_iocs

