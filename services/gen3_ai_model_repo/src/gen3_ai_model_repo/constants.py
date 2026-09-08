"""
Constants and security-related configurations for the model repository service.

This module defines default security scan status structures and other application constants.
"""

# Default security file status structure
# This represents the standard security scan metadata returned in API responses.
# Each scan type can have a status and additional metadata.
#
# Currently, all scans default to "unscanned" as the scanning features are planned
# but not yet implemented. Future versions should integrate with:
# - JFrog Artifactory for artifact scanning
# - Protect AI for model security scanning
# - ClamAV or similar for antivirus scanning
# - Pickle import analysis tools
# - VirusTotal API for file scanning
#
# See: https://huggingface.co/docs/hub/security for reference on HF security standards
DEFAULT_SECURITY_FILE_STATUS = {
    "status": "unscanned",
    "jFrogScan": {"status": "unscanned"},
    "protectAiScan": {"status": "unscanned"},
    "avScan": {"status": "unscanned"},
    "pickleImportScan": {"status": "unscanned"},
    "virusTotalScan": {"status": "unscanned"},
}
