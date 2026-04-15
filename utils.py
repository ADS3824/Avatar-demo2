"""
utils.py — Utility functions for ResumeIQ

Handles:
- Azure Search queries
- SAS URL generation
- Data parsing and extraction
- Logging and error handling
"""

import os
import json
import logging
import urllib.parse
import requests
from datetime import datetime, timezone, timedelta
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# AZURE BLOB STORAGE — SAS URLs
# ══════════════════════════════════════════════════════════════════

def list_blobs_in_container() -> list:
    """List all blobs in the container for debugging."""
    try:
        from azure.storage.blob import ContainerClient
        account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        container = os.environ.get("AZURE_STORAGE_CONTAINER_NAME")
        
        if not all([account, key, container]):
            return []
        
        container_client = ContainerClient(
            account_url=f"https://{account}.blob.core.windows.net",
            container_name=container,
            credential=key
        )
        
        blobs = [blob.name for blob in container_client.list_blobs()]
        logger.info(f"Container has {len(blobs)} blobs available")
        return blobs
    except Exception as e:
        logger.error(f"Failed to list blobs: {e}")
        return []


def find_matching_blob(file_name: str) -> str:
    """Find the actual blob name in storage that matches the given file_name.
    
    Handles cases where agent returns slightly different names than what's in storage.
    """
    if not file_name:
        return ""
    
    try:
        from azure.storage.blob import ContainerClient
        account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        container = os.environ.get("AZURE_STORAGE_CONTAINER_NAME")
        
        if not all([account, key, container]):
            return ""
        
        # Decode if URL-encoded
        decoded_name = urllib.parse.unquote(file_name) if "%" in file_name else file_name
        
        container_client = ContainerClient(
            account_url=f"https://{account}.blob.core.windows.net",
            container_name=container,
            credential=key
        )
        
        # First, try exact match
        for blob in container_client.list_blobs():
            if blob.name == decoded_name:
                logger.info(f"✅ Exact match found: {decoded_name}")
                return decoded_name
        
        # If no exact match, try case-insensitive match
        decoded_lower = decoded_name.lower()
        for blob in container_client.list_blobs():
            if blob.name.lower() == decoded_lower:
                logger.info(f"✅ Case-insensitive match found: {blob.name}")
                return blob.name
        
        logger.warning(f"❌ No blob found matching: {decoded_name}")
        return ""
        
    except Exception as e:
        logger.error(f"Failed to find matching blob: {e}")
        return ""

def generate_sas_url(blob_name: str, expiry_hours: int = 2, validate_exists: bool | None = None) -> str:
    """Generate a secure SAS URL for a blob with 2-hour expiry.

    When validate_exists is False (default), skips storage listing and generates
    a SAS URL even if the blob does not exist.
    """
    if not blob_name:
        logger.warning("generate_sas_url: blob_name is empty")
        return ""

    try:
        account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        container = os.environ.get("AZURE_STORAGE_CONTAINER_NAME")

        if not all([account, key, container]):
            logger.error(f"Azure Storage env vars missing: account={bool(account)}, key={bool(key)}, container={bool(container)}")
            return ""

        if validate_exists is None:
            validate_exists = os.environ.get("RESUME_VALIDATE_BLOBS", "0").strip().lower() in ("1", "true", "yes")

        # Decode if URL-encoded
        decoded_name = urllib.parse.unquote(blob_name) if "%" in blob_name else blob_name

        if validate_exists:
            # Try to find the actual blob name in storage (handles naming mismatches)
            actual_blob_name = find_matching_blob(decoded_name)
            if not actual_blob_name:
                logger.error(f"❌ Could not find blob in storage: {decoded_name}")
                # Debug: show what is actually in the container
                available = list_blobs_in_container()
                if available:
                    logger.info(f"Available blobs (first 10): {available[:10]}")
                return ""
        else:
            actual_blob_name = decoded_name

        # Generate SAS token using actual blob name
        token = generate_blob_sas(
            account_name=account,
            container_name=container,
            blob_name=actual_blob_name,
            account_key=key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
        )

        # URL-encode the blob name for the final URL
        encoded_name = urllib.parse.quote(actual_blob_name, safe="")
        url = f"https://{account}.blob.core.windows.net/{container}/{encoded_name}?{token}"

        logger.info(f"✅ SAS URL generated for: {actual_blob_name} (expiry: {expiry_hours}h)")
        return url

    except Exception as e:
        logger.error(f"SAS URL generation failed for {blob_name}: {e}", exc_info=True)
        return ""

# ══════════════════════════════════════════════════════════════════
# DATA PARSING & HELPERS
# ══════════════════════════════════════════════════════════════════

def safe_list(val) -> list:
    """Safely convert any value to a list."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, ValueError):
            return [val] if val.strip() else []
    return [val]


def parse_json_response(text: str):
    """Parse JSON from agent response, stripping markdown if present."""
    clean = text.strip()
    
    # Try stripping markdown code fences
    if "```" in clean:
        parts = clean.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("[") or p.startswith("{"):
                clean = p
                break
    
    return json.loads(clean)


def extract_name_from_filename(fname: str) -> str:
    """Derive a readable candidate name from filename."""
    import re
    # Remove extension
    stem = re.sub(r"\.[a-zA-Z]{2,5}$", "", fname)
    # Replace separators with spaces
    stem = re.sub(r"[_\-\d\[\]()\s]+", " ", stem).strip()
    return stem.title() if stem else fname


# ══════════════════════════════════════════════════════════════════
# VALIDATION & NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def ensure_required_fields(candidate: dict) -> dict:
    """Ensure all required Phase 1 fields exist with fallback values."""
    required = {
        "candidate_id": "",
        "file_name": "unknown.pdf",
        "email": "N/A",
        "phone": "N/A",
        "current_title": "N/A",
        "highest_education": "N/A",
        "total_experience_years": "N/A",
        "current_location": "N/A",
        "notice_period": "N/A",
        "skills": [],
        "certifications": [],
        "brief": "No summary available.",
    }
    for key, default in required.items():
        if key not in candidate or candidate[key] is None:
            candidate[key] = default
        # Coerce skills/certifications to lists
        if key in ("skills", "certifications"):
            candidate[key] = safe_list(candidate[key])
    return candidate


def ensure_screening_fields(result: dict) -> dict:
    """Ensure all Phase 2 result fields exist."""
    required = {
        "file_name": "unknown.pdf",
        "score": 0,
        "reasons": "No details available.",
        "gaps": "",
    }
    for key, default in required.items():
        if key not in result or result[key] is None:
            result[key] = default
    # Ensure score is int
    try:
        result["score"] = int(result["score"])
    except (ValueError, TypeError):
        result["score"] = 0
    return result


# ══════════════════════════════════════════════════════════════════
# METRICS CALCULATION
# ══════════════════════════════════════════════════════════════════

def compute_metrics(df) -> dict:
    """Calculate aggregate metrics from candidate dataframe."""
    if df is None or len(df) == 0:
        return {
            "total": 0,
            "unique_titles": 0,
            "unique_locations": 0,
            "unique_skills": 0,
        }
    
    all_skills = []
    for skills in df.get("skills", []):
        all_skills.extend(safe_list(skills))
    
    return {
        "total": len(df),
        "unique_titles": df.get("current_title", []).nunique() if "current_title" in df.columns else 0,
        "unique_locations": df.get("current_location", []).nunique() if "current_location" in df.columns else 0,
        "unique_skills": len(set(all_skills)),
    }


# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)
