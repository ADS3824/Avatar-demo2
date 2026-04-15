"""
foundry.py - Azure AI Foundry agent client.

Calls published Foundry agents through the Responses REST API using
service-principal credentials from environment variables.
"""

import os
import json
import logging
import time
import requests
import pandas as pd
import sqlite3
from utils import parse_json_response, ensure_required_fields, ensure_screening_fields, generate_sas_url, safe_list

logger = logging.getLogger(__name__)


class FoundryClient:
    """Client for calling Foundry agents using Foundry Responses REST API."""

    def __init__(self):
        self.endpoint = (
            os.environ.get("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT")
            or os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
        )
        if not self.endpoint:
            raise RuntimeError("AZURE_AI_PROJECT_ENDPOINT or AZURE_AI_FOUNDRY_PROJECT_ENDPOINT not set")
        self.endpoint = self.endpoint.rstrip("/")

        self.tenant_id     = os.environ.get("AZURE_TENANT_ID")
        self.client_id     = os.environ.get("AZURE_CLIENT_ID")
        self.client_secret = os.environ.get("AZURE_CLIENT_SECRET")
        if not self.tenant_id or not self.client_id or not self.client_secret:
            raise RuntimeError("AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET must be set")

        self.tabulation_agent   = os.environ.get("FOUNDRY_TABULATION_AGENT_NAME", "resumetabulation")
        self.tabulation_version = os.environ.get("FOUNDRY_TABULATION_VERSION",    "26")
        self.screening_agent    = os.environ.get("FOUNDRY_SCREENING_AGENT_NAME",  "resumescreening")
        self.screening_version  = os.environ.get("FOUNDRY_SCREENING_VERSION",     "17")
        self.model_name         = (
            os.environ.get("AZURE_FOUNDRY_MODEL_NAME")
            or os.environ.get("FOUNDRY_MODEL_NAME")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or "gpt-4.1-mini"
        )
        self.api_version        = os.environ.get("FOUNDRY_API_VERSION") or os.environ.get("AZURE_FOUNDRY_API_VERSION") or "2025-11-15-preview"

        self._cached_token = None
        self._token_expiry = 0

        self.cache_db  = os.environ.get("RESUME_CACHE_DB",  ".resume_cache.db")
        self.cache_ttl = int(os.environ.get("RESUME_CACHE_TTL", str(60 * 60)))
        self._init_cache_db()

        logger.info(f"Agent versions: tabulation={self.tabulation_version}, screening={self.screening_version}")

    # ── Auth ──────────────────────────────────────────────────────
    def _get_access_token(self) -> str:
        now = int(time.time())
        if self._cached_token and now < (self._token_expiry - 60):
            return self._cached_token

        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        response  = requests.post(
            token_url,
            data={
                "grant_type":    "client_credentials",
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
                "scope":         "https://ai.azure.com/.default",
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        self._cached_token = payload["access_token"]
        self._token_expiry = now + int(payload.get("expires_in", 3600))
        return self._cached_token

    # ── Response parser ───────────────────────────────────────────
    @staticmethod
    def _extract_output_text(payload: dict) -> str:
        def extract_text_from_content(content):
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, dict):
                text_val = content.get("text") or content.get("value") or content.get("message")
                if isinstance(text_val, str):
                    return text_val.strip()
                nested = []
                for nested_content in content.get("content", []) or []:
                    nested_text = extract_text_from_content(nested_content)
                    if nested_text:
                        nested.append(nested_text)
                return "\n".join(nested).strip()
            return ""

        def detect_tool_approval(items):
            if isinstance(items, dict):
                items = [items]
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type") or ""
                if item_type in ("mcp_list_tools", "mcp_approval_request"):
                    return item
            return None

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        pieces = []
        for source_key in ("output", "data", "content"):
            items = payload.get(source_key) or []
            if isinstance(items, dict):
                items = [items]
            for item in items:
                if isinstance(item, str):
                    pieces.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                content = item.get("content") or item.get("tool_output") or item.get("message")
                if content is None and source_key == "content":
                    content = item
                if isinstance(content, list):
                    for piece in content:
                        text = extract_text_from_content(piece)
                        if text:
                            pieces.append(text)
                else:
                    text = extract_text_from_content(content)
                    if text:
                        pieces.append(text)

        if pieces:
            return "\n".join([p for p in pieces if p])

        tool_item = detect_tool_approval(payload.get("output")) or detect_tool_approval(payload.get("data")) or detect_tool_approval(payload.get("content"))
        if tool_item is not None:
            arguments = tool_item.get("arguments")
            query_details = []
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = None
            if isinstance(arguments, dict):
                queries = arguments.get("queries")
                if isinstance(queries, list):
                    query_details = [str(q).strip() for q in queries if str(q).strip()]
            if query_details:
                query_text = "; ".join(query_details)
                return (
                    "The agent requested an approval-stage tool call for these queries: "
                    f"{query_text}. "
                    "The tool requires approval before producing the final answer. "
                    "Please configure the agent/tool to auto-approve or use a direct retrieval setup."
                )
            return (
                "The agent requested a tool call and is waiting for approval before the final answer can be returned. "
                "Please configure the agent/tool to auto-approve or use a direct retrieval setup."
            )

        if isinstance(payload, dict):
            for key in ("status", "id", "model"):  # preserve useful metadata only
                payload.pop(key, None)
            return json.dumps(payload, ensure_ascii=False)
        return ""

    # ── Agent caller ──────────────────────────────────────────────
    def _call_agent(self, agent_name: str, version: str, message: str) -> str:
        """Call a Foundry published agent using the Responses REST API."""
        logger.info(f"Calling Foundry agent: {agent_name} v{version}")

        token   = self._get_access_token()
        url     = f"{self.endpoint}/openai/responses?api-version={self.api_version}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }
        body = {
            "model": self.model_name,
            "input": [{"role": "user", "content": message}],
            "stream": False,
            "agent": {
                "type":    "agent_reference",
                "name":    agent_name,
                "version": version,
            },
            "tool_choice": "auto",
        }
        response = requests.post(url, headers=headers, json=body, timeout=120)
        if not response.ok:
            try:
                body_text = response.text
            except Exception:
                body_text = "<unable to read response body>"
            logger.error(
                "Foundry Responses API returned error %s: %s",
                response.status_code,
                body_text,
            )
        response.raise_for_status()
        output_text = self._extract_output_text(response.json())
        logger.info(f"Agent response length: {len(output_text)} chars")
        return output_text

    def _enrich_missing_fields(self, candidates: list) -> list:
        """Use the model to extract missing fields from resume content."""
        enable = os.environ.get("RESUME_ENRICH_MISSING", "1").strip().lower() not in ("0", "false", "no")
        if not enable:
            return candidates

        max_chars = int(os.environ.get("RESUME_ENRICH_MAX_CHARS", "5000"))
        batch_size = int(os.environ.get("RESUME_ENRICH_BATCH", "8"))

        def needs_enrich(c: dict) -> bool:
            fields = (
                "email",
                "phone",
                "current_title",
                "highest_education",
                "total_experience_years",
                "current_location",
                "skills",
                "certifications",
            )
            for f in fields:
                v = c.get(f)
                if v in (None, "", "N/A", []):
                    return True
            return False

        def pick_text(c: dict) -> str:
            for k in ("content", "raw_content", "brief"):
                v = c.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        # Prepare enrichment list
        pending = []
        for c in candidates:
            if not needs_enrich(c):
                continue
            text = pick_text(c)
            if not text:
                continue
            uid = (
                str(c.get("candidate_id") or c.get("id") or c.get("parent_id") or c.get("file_name") or "").strip()
            )
            if not uid:
                continue
            pending.append({
                "uid": uid,
                "file_name": c.get("file_name", ""),
                "content": text[:max_chars],
            })

        if not pending:
            return candidates

        logger.info("Enriching %s candidates via tabulation agent", len(pending))

        by_uid = {str(c.get("candidate_id") or c.get("id") or c.get("parent_id") or c.get("file_name") or "").strip(): c for c in candidates}

        # Batch enrichment
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            prompt = (
                "Extract resume fields from the following resume texts. "
                "Return ONLY a JSON array.\n\n"
                "For each item, return:\n"
                "- uid (same as input)\n"
                "- file_name (best guess; keep input if unknown)\n"
                "- email, phone, current_title, highest_education, total_experience_years,\n"
                "  current_location, notice_period, skills (list), certifications (list), brief\n"
                "If a field is missing, use \"N/A\" (or [] for lists).\n\n"
                f"INPUT:\n{json.dumps(batch, ensure_ascii=False)}"
            )
            raw = self._call_agent(self.tabulation_agent, self.tabulation_version, prompt)
            try:
                parsed = parse_json_response(raw)
            except Exception as e:
                logger.warning(f"Failed to parse enrichment response: {e}")
                continue
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                logger.warning("Unexpected enrichment response type: %s", type(parsed))
                continue
            for r in parsed:
                uid = str(r.get("uid", "")).strip()
                if not uid:
                    continue
                target = by_uid.get(uid)
                if not target:
                    continue
                # Merge non-empty fields
                for k, v in r.items():
                    if k == "uid":
                        continue
                    if v not in (None, "", "N/A", [], {}):
                        target[k] = v

        return candidates

    # ── Azure AI Search fetch ────────────────────────────────────
    def _fetch_all_from_search(self, batch_size: int = 1000) -> list:
        """Fetch all documents directly from Azure AI Search."""
        endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        api_key  = os.environ.get("AZURE_SEARCH_API_KEY")
        index    = os.environ.get("AZURE_SEARCH_INDEX_NAME")
        if not all([endpoint, api_key, index]):
            return []

        api_version = os.environ.get("AZURE_SEARCH_API_VERSION", "2023-11-01")
        url = f"{endpoint.rstrip('/')}/indexes/{index}/docs/search?api-version={api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }

        all_docs = []
        skip = 0
        while True:
            body = {
                "search": "*",
                "top": int(batch_size),
                "skip": int(skip),
            }
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            docs = payload.get("value", []) or []
            if not docs:
                break
            all_docs.extend(docs)
            skip += len(docs)
            if len(docs) < batch_size:
                break

        logger.info("Azure Search returned %s candidates", len(all_docs))
        return all_docs

    @staticmethod
    def _candidate_key(candidate: dict) -> str:
        """Stable key for deduping candidates."""
        for k in ("file_name", "fileName", "filename", "blob_name", "blobName", "name"):
            v = str(candidate.get(k, "")).strip()
            if v and v.lower() not in ("unknown.pdf", "unknown"):
                return f"file:{v.lower()}"
        for k in ("candidate_id", "candidateId", "id"):
            v = str(candidate.get(k, "")).strip()
            if v:
                return f"id:{v}"
        email = str(candidate.get("email", "")).strip().lower()
        if email:
            return f"email:{email}"
        return f"raw:{hash(json.dumps(candidate, sort_keys=True, ensure_ascii=False))}"

    @staticmethod
    def _normalize_candidate(candidate: dict) -> dict:
        """Normalize common field names to expected schema."""
        c = dict(candidate)
        def first_non_empty(keys):
            for k in keys:
                v = c.get(k)
                if isinstance(v, list) and v:
                    return v[0]
                if v not in (None, "", "N/A"):
                    return v
            return None

        # file name normalization
        file_name = first_non_empty(
            ("file_name", "fileName", "filename", "blob_name", "blobName", "name", "metadata_storage_name")
        )
        if not file_name:
            path = first_non_empty(("metadata_storage_path", "metadata_storage_path_encoded", "path"))
            if path and isinstance(path, str):
                try:
                    from urllib.parse import unquote
                    file_name = unquote(path.split("/")[-1])
                except Exception:
                    file_name = path.split("/")[-1]
        if file_name:
            c["file_name"] = str(file_name).strip()

        # candidate id normalization
        candidate_id = first_non_empty(("candidate_id", "candidateId", "id", "metadata_storage_path"))
        if candidate_id:
            c["candidate_id"] = str(candidate_id).strip()

        # field mapping from common search schemas
        field_map = {
            "email": ("email", "emails", "email_address", "emailAddress"),
            "phone": ("phone", "phone_number", "phoneNumber", "mobile", "mobile_number"),
            "current_title": ("current_title", "currentTitle", "title", "current_role", "currentRole", "job_title", "designation"),
            "highest_education": ("highest_education", "highestEducation", "education", "education_level", "highestQualification"),
            "total_experience_years": ("total_experience_years", "totalExperienceYears", "experience_years", "years_of_experience", "total_experience", "experience"),
            "current_location": ("current_location", "currentLocation", "location", "current_city", "city"),
            "notice_period": ("notice_period", "noticePeriod", "np"),
            "skills": ("skills", "skillset", "skill_set", "key_skills", "skills_list"),
            "certifications": ("certifications", "certs", "certificates"),
            "brief": ("brief", "summary", "profile", "resume_summary"),
        }
        for target, keys in field_map.items():
            if c.get(target) in (None, "", "N/A", []):
                val = first_non_empty(keys)
                if val is not None:
                    c[target] = val

        # Heuristic fallback: match by key substrings if still missing
        if c.get("email") in (None, "", "N/A"):
            for k in c.keys():
                if "email" in k.lower():
                    c["email"] = c.get(k)
                    break
        if c.get("phone") in (None, "", "N/A"):
            for k in c.keys():
                lk = k.lower()
                if "phone" in lk or "mobile" in lk:
                    c["phone"] = c.get(k)
                    break
        if c.get("current_title") in (None, "", "N/A"):
            for k in c.keys():
                if "title" in k.lower():
                    c["current_title"] = c.get(k)
                    break
        if c.get("highest_education") in (None, "", "N/A"):
            for k in c.keys():
                if "education" in k.lower() or "qualification" in k.lower():
                    c["highest_education"] = c.get(k)
                    break
        if c.get("total_experience_years") in (None, "", "N/A"):
            for k in c.keys():
                if "experience" in k.lower():
                    c["total_experience_years"] = c.get(k)
                    break
        if c.get("current_location") in (None, "", "N/A"):
            for k in c.keys():
                lk = k.lower()
                if "location" in lk or "city" in lk:
                    c["current_location"] = c.get(k)
                    break
        if c.get("skills") in (None, "", "N/A", []):
            for k in c.keys():
                if "skill" in k.lower():
                    c["skills"] = c.get(k)
                    break
        if c.get("certifications") in (None, "", "N/A", []):
            for k in c.keys():
                lk = k.lower()
                if "cert" in lk:
                    c["certifications"] = c.get(k)
                    break
        if c.get("brief") in (None, "", "N/A"):
            for k in c.keys():
                lk = k.lower()
                if "summary" in lk or "brief" in lk or "profile" in lk:
                    c["brief"] = c.get(k)
                    break

        # If brief still missing, use a trimmed content field if present
        if c.get("brief") in (None, "", "N/A"):
            content = first_non_empty(("content", "merged_content", "text"))
            if isinstance(content, str) and content.strip():
                c["brief"] = content.strip()[:500]

        # Normalize common string fields (trim extra whitespace)
        for k in ("file_name", "email", "phone", "current_title", "highest_education", "current_location", "notice_period", "brief"):
            v = c.get(k)
            if isinstance(v, str):
                c[k] = " ".join(v.split()).strip()
        return c
        return c

    # ── SQLite cache ──────────────────────────────────────────────
    def _init_cache_db(self) -> None:
        try:
            conn = sqlite3.connect(self.cache_db)
            cur  = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS candidates "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)"
            )
            conn.commit()
        finally:
            conn.close()

    def _get_cache_timestamp(self) -> int:
        try:
            conn = sqlite3.connect(self.cache_db)
            cur  = conn.cursor()
            cur.execute("SELECT v FROM meta WHERE k='last_updated'")
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _load_cached_candidates(self) -> list:
        try:
            conn = sqlite3.connect(self.cache_db)
            cur  = conn.cursor()
            cur.execute("SELECT data FROM candidates ORDER BY id")
            rows = cur.fetchall()
            return [json.loads(r[0]) for r in rows]
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _save_candidates_to_cache(self, candidates: list) -> None:
        try:
            conn = sqlite3.connect(self.cache_db)
            cur  = conn.cursor()
            cur.execute("DELETE FROM candidates")
            for c in candidates:
                cur.execute(
                    "INSERT INTO candidates (data) VALUES (?)",
                    (json.dumps(c, ensure_ascii=False),),
                )
            ts = int(time.time())
            cur.execute(
                "INSERT OR REPLACE INTO meta (k, v) VALUES ('last_updated', ?)",
                (str(ts),),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ── Fetch all resumes ─────────────────────────────────────────
    def fetch_all_resumes(self, force_refresh: bool = False, min_expected: int | None = None) -> pd.DataFrame:
        """Fetch all resumes directly from tabulation agent.

        force_refresh: bypass local cache even if within TTL.
        min_expected: if set, cache is ignored when it has fewer candidates than this value.
        """
        logger.info(f"Fetching all resumes from tabulation agent v{self.tabulation_version}...")

        try:
            if min_expected is None:
                min_expected = int(os.environ.get("RESUME_MIN_CANDIDATES", "0"))

            sync_always = os.environ.get("RESUME_SYNC_ALWAYS", "1").strip().lower() not in ("0", "false", "no")

            cached = self._load_cached_candidates()
            cached_map = {}
            for c in cached:
                key = self._candidate_key(c)
                cached_map[key] = c

            last_ts = self._get_cache_timestamp()
            cache_fresh = last_ts and (time.time() - last_ts) < self.cache_ttl
            if not force_refresh and not sync_always and cache_fresh and cached:
                if min_expected and len(cached) < min_expected:
                    logger.info(
                        "Cache has %s candidates, below min_expected=%s; bypassing cache",
                        len(cached),
                        min_expected,
                    )
                else:
                    logger.info("Loading candidates from local cache (within TTL)")
                    all_candidates = [ensure_required_fields(c) for c in cached]
                    df = pd.DataFrame(all_candidates)
                    logger.info(f"Loaded {len(df)} candidates from cache")
                    return df
            if force_refresh:
                logger.info("Force refresh enabled; bypassing cache")

            parsed = []
            from_search = False
            try:
                batch_size = int(os.environ.get("RESUME_SEARCH_BATCH_SIZE", "1000"))
                parsed = self._fetch_all_from_search(batch_size=batch_size)
                from_search = bool(parsed)
            except Exception as e:
                logger.warning(f"Search fetch failed, falling back to agent: {e}")

            if not parsed:
                prompt = (
                    "RETRIEVE ALL CANDIDATES FROM THE RESUME INDEX.\n\n"
                    "Instructions:\n"
                    "1. Make MULTIPLE Azure AI Search queries with different keywords "
                    "(skills, experience, education, titles, locations)\n"
                    "2. Retrieve candidates in batches of up to 50\n"
                    "3. Continue querying until you have exhausted all results\n"
                    "4. Deduplicate by file_name\n"
                    "5. For EACH candidate, extract ALL fields:\n"
                    "   - candidate_id, file_name, email, phone, current_title,\n"
                    "   - highest_education, total_experience_years, current_location,\n"
                    "   - notice_period, skills (list), certifications (list), brief\n\n"
                    "Output: Return ONLY a JSON array with ALL candidates found.\n"
                    "NO explanations, NO markdown, NO other text."
                )

                raw    = self._call_agent(self.tabulation_agent, self.tabulation_version, prompt)
                parsed = parse_json_response(raw)
                from_search = False

            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                if "raw" in locals():
                    logger.warning(f"Agent returned non-list type: {type(parsed)}, raw: {raw[:200]}")
                else:
                    logger.warning(f"Search returned non-list type: {type(parsed)}")
                return pd.DataFrame()
            if not parsed:
                logger.warning("Agent returned empty list")
                if cached:
                    all_candidates = [ensure_required_fields(c) for c in cached]
                    df = pd.DataFrame(all_candidates)
                    logger.info(f"Loaded {len(df)} candidates from cache (fallback)")
                    return df
                return pd.DataFrame()

            max_candidates = int(os.environ.get("RESUME_MAX_CANDIDATES", "0"))
            if max_candidates and len(parsed) > max_candidates:
                logger.info("Limiting candidates to %s (test mode)", max_candidates)
                parsed = parsed[:max_candidates]

            source_label = "search" if from_search else "agent"
            logger.info(f"Fetched {len(parsed)} candidates from {source_label}")

            normalized = [ensure_required_fields(self._normalize_candidate(c)) for c in parsed]

            new_count = 0
            for c in normalized:
                key = self._candidate_key(c)
                if key not in cached_map:
                    new_count += 1
                    cached_map[key] = c
                else:
                    # Merge: prefer non-empty new values, otherwise keep cached
                    existing = cached_map[key]
                    merged = dict(existing)
                    for k, v in c.items():
                        if v not in (None, "", "N/A", [], {}):
                            merged[k] = v
                    cached_map[key] = merged

            merged = list(cached_map.values())

            # Enrich missing fields from content using the model
            merged = self._enrich_missing_fields(merged)

            # Normalize/ensure required fields after enrichment
            merged = [ensure_required_fields(self._normalize_candidate(c)) for c in merged]

            try:
                self._save_candidates_to_cache(merged)
                logger.info("Saved candidates to local cache")
            except Exception:
                logger.warning("Could not cache candidates")

            logger.info("Cache now has %s candidates (%s new)", len(merged), new_count)

            for candidate in merged:
                url = generate_sas_url(candidate.get("file_name", ""))
                candidate["preview_url"]  = url
                candidate["download_url"] = url

            df = pd.DataFrame(merged)
            logger.info(f"Loaded {len(df)} candidates into DataFrame")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch resumes: {e}")
            return pd.DataFrame()

    # ── Screen resumes ────────────────────────────────────────────
    def screen_resumes(self, jd_text: str, candidates_df: pd.DataFrame, min_score: int = 0, top_n: int = 10) -> list:
        """Call screening agent to score candidates against JD."""
        logger.info(f"Screening {len(candidates_df)} candidates...")

        try:
            # FIX 3: direct column access
            if "file_name" not in candidates_df.columns or len(candidates_df) == 0:
                logger.warning("No candidates in DataFrame")
                return []

            candidate_files = candidates_df["file_name"].tolist()
            logger.info(f"Candidates to screen: {candidate_files}")

            # FIX 1: send full profiles instead of just filenames
            candidate_profiles = []
            for _, row in candidates_df.iterrows():
                profile = {
                    "file_name":              row.get("file_name",              ""),
                    "current_title":          row.get("current_title",          "N/A"),
                    "highest_education":      row.get("highest_education",      "N/A"),
                    "total_experience_years": row.get("total_experience_years", "N/A"),
                    "current_location":       row.get("current_location",       "N/A"),
                    "notice_period":          row.get("notice_period",          "N/A"),
                    "skills":                 safe_list(row.get("skills",         [])),
                    "certifications":         safe_list(row.get("certifications", [])),
                    "brief":                  row.get("brief",                  ""),
                }
                candidate_profiles.append(profile)

            prompt = (
                f"Score each candidate resume profile against the job description below.\n\n"
                f"CANDIDATE PROFILES:\n"
                f"{json.dumps(candidate_profiles, indent=2)}\n\n"
                f"JOB DESCRIPTION:\n"
                f"{jd_text}\n\n"
                f"Instructions:\n"
                f"1. Score EVERY candidate listed above. Do not skip any.\n"
                f"2. For each candidate return:\n"
                f"   - file_name (exactly as given, do not change the casing)\n"
                f"   - score (integer 0-100)\n"
                f"   - reasons (list of strings — why they match)\n"
                f"   - gaps (list of strings — what they are missing)\n"
                f"3. Return ONLY a valid JSON array. No markdown, no explanation, no extra text.\n"
                f"Example format:\n"
                f'[{{"file_name":"x.pdf","score":82,"reasons":["..."],"gaps":["..."]}}]'
            )

            raw = self._call_agent(self.screening_agent, self.screening_version, prompt)
            logger.info(f"Agent response (first 500 chars): {raw[:500]}")

            parsed = parse_json_response(raw)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                logger.warning(f"Unexpected response type: {type(parsed)}")
                return []

            logger.info(f"Parsed {len(parsed)} results from agent")

            results = [ensure_screening_fields(r) for r in parsed]

            # FIX 2: case-insensitive merge so file_name mismatches don't silently drop candidates
            candidates_map = {
                str(r.get("file_name", "")).strip().lower(): dict(r)
                for _, r in candidates_df.iterrows()
            }

            merged = []
            for result in results:
                fname       = str(result.get("file_name", "")).strip()
                fname_lower = fname.lower()

                candidate = candidates_map.get(fname_lower)

                if candidate is None:
                    logger.warning(f"No match found for agent-returned name: '{fname}' — skipping")
                    continue

                merged_result               = {**candidate, **result}
                merged_result["file_name"]  = candidate.get("file_name", fname)   # restore original casing
                merged_result["preview_url"]  = generate_sas_url(merged_result["file_name"])
                merged_result["download_url"] = merged_result["preview_url"]
                merged.append(merged_result)

            logger.info(f"After merge: {len(merged)} / {len(results)} results matched")

            if len(merged) < len(results):
                logger.warning(
                    f"{len(results) - len(merged)} candidates dropped. "
                    f"Agent names: {[r.get('file_name') for r in results]} | "
                    f"Map keys: {list(candidates_map.keys())}"
                )

            merged.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
            filtered = [r for r in merged if int(r.get("score", 0)) >= min_score]

            logger.info(f"Final results (min_score={min_score}): {len(filtered)}")
            return filtered[:int(top_n)]

        except Exception as e:
            logger.error(f"Screening failed: {e}", exc_info=True)
            return []
