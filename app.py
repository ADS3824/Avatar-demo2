import os
import re
import json
import logging
import tempfile
import requests
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.communication.identity import CommunicationIdentityClient
from foundry import FoundryClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = {
    'openai_key': os.getenv('AZURE_OPENAI_KEY'),
    'openai_endpoint': (os.getenv('AZURE_OPENAI_ENDPOINT') or '').rstrip('/'),
    'openai_api_version': os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'),
    'whisper_deployment': (
        os.getenv('WHISPER_DEPLOYMENT')
        or os.getenv('AZURE_WHISPER_DEPLOYMENT')
        or os.getenv('AZURE_OPENAI_DEPLOYMENT')
    ),
    'speech_key': os.getenv('AZURE_SPEECH_KEY'),
    'speech_region': os.getenv('AZURE_SPEECH_REGION'),
    'speech_endpoint': (os.getenv('AZURE_SPEECH_ENDPOINT') or '').rstrip('/'),
    'acs_connection_string': os.getenv('ACS_CONNECTION_STRING'),
    'agent_reference': os.getenv('AZURE_AGENT_ID') or os.getenv('AZURE_EXISTING_AGENT_ID') or '',
    'agent_name': os.getenv('AZURE_AGENT_NAME') or os.getenv('FOUNDRY_AGENT_NAME'),
    'agent_version': os.getenv('AZURE_AGENT_VERSION') or os.getenv('FOUNDRY_AGENT_VERSION'),
    'foundry_endpoint': (
        os.getenv('AZURE_AI_PROJECT_ENDPOINT')
        or os.getenv('AZURE_AI_FOUNDRY_PROJECT_ENDPOINT')
        or ''
    ).rstrip('/'),
    'foundry_base_model': (
        os.getenv('AZURE_FOUNDRY_MODEL_NAME')
        or os.getenv('FOUNDRY_MODEL_NAME')
        or os.getenv('AZURE_OPENAI_DEPLOYMENT')
        or 'gpt-4.1-mini'
    ),
    'avatar_character': os.getenv('AVATAR_CHARACTER', 'jeff'),
    'avatar_style': os.getenv('AVATAR_STYLE', 'business'),
    'avatar_voice': os.getenv('VOICE_NAME', 'en-IN-ArjunNeural'),
    'wake_word': os.getenv('WAKE_WORD', 'Aravindan Sir'),
    'tenant_id': os.getenv('AZURE_TENANT_ID'),
    'client_id': os.getenv('AZURE_CLIENT_ID'),
    'client_secret': os.getenv('AZURE_CLIENT_SECRET'),
}

foundry_client = None
foundry_error = None
try:
    foundry_client = FoundryClient()
    logger.info('✅ Foundry client initialized')
except Exception as exc:
    foundry_error = str(exc)
    logger.warning('⚠️ Foundry client initialization failed: %s', foundry_error)


def get_agent_reference():
    name = config['agent_name']
    version = config['agent_version']
    if not name and config['agent_reference']:
        parts = config['agent_reference'].split(':', 1)
        if len(parts) == 2:
            name, version = parts[0].strip(), parts[1].strip()
        else:
            name = config['agent_reference'].strip()
    return name, version


def build_avatar_relay_urls(urls):
    if not urls:
        return []

    if isinstance(urls, str):
        urls = [urls]

    relay_mode = (os.getenv('AVATAR_RELAY_MODE', 'tcp443') or 'tcp443').strip().lower()
    custom_relay_urls = [u.strip() for u in (os.getenv('AVATAR_TURN_URLS') or '').split(',') if u.strip()]
    combined = [*custom_relay_urls, *[u for u in urls if isinstance(u, str) and u.strip()]]

    deduped = []
    seen = set()

    def push_url(url):
        if not url or url in seen:
            return
        seen.add(url)
        deduped.append(url)

    for url in combined:
        push_url(url)

    for url in list(deduped):
        match = re.match(r'^turns?:([^?]+)(\?.*)?$', url, re.IGNORECASE)
        if not match:
            continue
        host_port = match.group(1)
        query = match.group(2) or ''
        host = host_port.split(':')[0]
        if not host:
            continue
        push_url(f'turns:{host}:443?transport=tcp')
        if relay_mode == 'tcp':
            push_url(f'turn:{host}:3478?transport=tcp')
        if relay_mode == 'all' and 'transport=' not in query.lower():
            push_url(f'turn:{host}:3478?transport=udp')

    def priority(url):
        if re.match(r'^turns:.*transport=tcp', url, re.IGNORECASE):
            return 0
        if re.match(r'^turn:.*transport=tcp', url, re.IGNORECASE):
            return 1
        if re.match(r'^turns:', url, re.IGNORECASE):
            return 2
        if re.match(r'^turn:', url, re.IGNORECASE):
            return 3
        return 4

    sorted_urls = sorted(deduped, key=priority)
    secure_tcp_relay_only = [url for url in sorted_urls if re.match(r'^turns:.*:443\?transport=tcp', url, re.IGNORECASE)]
    return secure_tcp_relay_only if secure_tcp_relay_only else sorted_urls


def get_avatar_ice_config():
    region = config['speech_region']
    key = config['speech_key']
    if not region or not key:
        raise RuntimeError('Missing Azure Speech region or key for avatar ICE token.')

    url = f'https://{region}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1'
    response = requests.get(url, headers={'Ocp-Apim-Subscription-Key': key}, timeout=20)
    if not response.ok:
        raise RuntimeError(f'Avatar ICE request failed ({response.status_code}): {response.text}')

    payload = response.json()
    urls = payload.get('urls') or payload.get('Urls') or []
    username = payload.get('username') or payload.get('Username') or payload.get('iceServers', [{}])[0].get('username')
    credential = (
        payload.get('credential')
        or payload.get('Credential')
        or payload.get('password')
        or payload.get('Password')
        or payload.get('iceServers', [{}])[0].get('credential')
    )

    if not urls or not username or not credential:
        raise RuntimeError('Avatar ICE response did not include usable TURN configuration.')

    return {
        'urls': build_avatar_relay_urls(urls),
        'username': username,
        'credential': credential,
    }


def get_acs_token():
    connection_string = config['acs_connection_string']
    if not connection_string:
        raise RuntimeError('Missing ACS_CONNECTION_STRING for ACS token generation.')

    client = CommunicationIdentityClient(connection_string)
    user = client.create_user()
    token = client.get_token(user, scopes=['voip'])
    return {
        'user': {'communicationUserId': user.communication_user_id},
        'token': token.token,
    }


def get_speech_token():
    region = config['speech_region']
    key = config['speech_key']
    if not region or not key:
        raise RuntimeError('Missing Azure Speech region or key for speech token generation.')

    url = f'https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken'
    response = requests.post(url, headers={'Ocp-Apim-Subscription-Key': key}, timeout=20)
    if not response.ok:
        raise RuntimeError(f'Speech token request failed ({response.status_code}): {response.text}')

    return {
        'token': response.text,
        'region': region,
        'voice': config['avatar_voice'],
        'avatarCharacter': config['avatar_character'],
        'avatarStyle': config['avatar_style'],
    }


def get_azure_credential():
    tenant_id = config['tenant_id']
    client_id = config['client_id']
    client_secret = config['client_secret']
    if not tenant_id or not client_id or not client_secret:
        raise RuntimeError('Missing Azure AD credentials for Foundry authentication.')
    return ClientSecretCredential(tenant_id, client_id, client_secret)


def get_transcription_deployment():
    deployment = config['whisper_deployment']
    if not deployment:
        raise RuntimeError('Missing Whisper deployment. Set WHISPER_DEPLOYMENT, AZURE_WHISPER_DEPLOYMENT, or AZURE_OPENAI_DEPLOYMENT.')
    return deployment


def get_foundry_access_token():
    credential = get_azure_credential()
    token = credential.get_token('https://ai.azure.com/.default')
    return token.token


def transcribe_audio_file(file_path: str) -> str:
    deployment = get_transcription_deployment()
    if config['foundry_endpoint']:
        endpoint = config['foundry_endpoint']
        url = f'{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version=2025-11-15-preview'
        headers = {'Authorization': f'Bearer {get_foundry_access_token()}'}
    else:
        endpoint = config['openai_endpoint']
        key = config['openai_key']
        if not endpoint or not key:
            raise RuntimeError('Missing Azure OpenAI endpoint or key for transcription.')
        url = f'{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version={config["openai_api_version"]}'
        headers = {'api-key': key}

    with open(file_path, 'rb') as handle:
        response = requests.post(url, headers=headers, files={'file': handle}, timeout=120)

    if not response.ok:
        raise RuntimeError(f'Transcription failed ({response.status_code}): {response.text}')

    payload = response.json()
    return payload.get('text') or payload.get('transcription') or ''


def extract_openai_response_text(payload):
    if not payload:
        return ''

    def extract_text_from_content(content):
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            text_val = content.get('text') or content.get('value') or content.get('message')
            if isinstance(text_val, str):
                return text_val.strip()
            nested = []
            for nested_content in content.get('content', []) or []:
                nested_text = extract_text_from_content(nested_content)
                if nested_text:
                    nested.append(nested_text)
            return '\n'.join(nested).strip()
        return ''

    def detect_tool_approval(items):
        if isinstance(items, dict):
            items = [items]
        for item in items or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get('type') or ''
            if item_type in ('mcp_list_tools', 'mcp_approval_request'):
                return item
        return None

    if isinstance(payload.get('output_text'), str) and payload['output_text'].strip():
        return payload['output_text'].strip()

    pieces = []
    for source_key in ('output', 'data', 'content'):
        items = payload.get(source_key) or []
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if isinstance(item, str):
                pieces.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            content = item.get('content') or item.get('tool_output') or item.get('message')
            if isinstance(content, list):
                for part in content:
                    text = extract_text_from_content(part)
                    if text:
                        pieces.append(text)
            else:
                text = extract_text_from_content(content)
                if text:
                    pieces.append(text)
            if not pieces and isinstance(item.get('text'), str):
                pieces.append(item.get('text').strip())

    if pieces:
        return '\n'.join([p for p in pieces if p]).strip()

    tool_item = detect_tool_approval(payload.get('output')) or detect_tool_approval(payload.get('data')) or detect_tool_approval(payload.get('content'))
    if tool_item is not None:
        arguments = tool_item.get('arguments')
        query_details = []
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if isinstance(arguments, dict):
            queries = arguments.get('queries')
            if isinstance(queries, list):
                query_details = [str(q).strip() for q in queries if str(q).strip()]
        if query_details:
            query_text = '; '.join(query_details)
            return (
                'The agent requested an approval-stage tool call for these queries: '
                f'{query_text}. '
                'The tool requires approval before the final answer can be returned. '
                'Please configure the agent/tool to auto-approve or use a direct retrieval setup.'
            )
        return (
            'The agent requested a tool call and is waiting for approval before the final answer can be returned. '
            'Please configure the agent/tool to auto-approve or use a direct retrieval setup.'
        )

    return ''


def get_agent_answer(question: str) -> str:
    if foundry_error:
        raise RuntimeError(foundry_error)
    if foundry_client is None:
        raise RuntimeError('Foundry client is not configured.')

    name, version = get_agent_reference()
    if not name or not version:
        raise RuntimeError('Missing Foundry agent name/version. Set AZURE_AGENT_NAME/FOUNDRY_AGENT_NAME and AZURE_AGENT_VERSION/FOUNDRY_AGENT_VERSION or AZURE_AGENT_ID=name:version.')

    return foundry_client._call_agent(name, version, question)


def contains_wake_word(text: str) -> bool:
    if not text:
        return False
    return config['wake_word'].strip().lower() in text.lower()


def process_question(audio_buffer: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
        tmp.write(audio_buffer)
        tmp_path = tmp.name

    try:
        text = transcribe_audio_file(tmp_path)
        if not contains_wake_word(text):
            logger.info('⏭️ Wake word not detected; skipping agent call.')
            return None

        cleaned = text.lower().replace(config['wake_word'].strip().lower(), '').strip()
        if not cleaned:
            answer = f'Please say {config["wake_word"]} followed by your question.'
            return {'question': '', 'answer': answer}

        answer = get_agent_answer(cleaned)
        return {'question': cleaned, 'answer': answer}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


logger.info('✅ Python app module loaded')
logger.info('🎯 Wake word: %s', config['wake_word'])
logger.info('👨 Avatar: %s (%s)', config['avatar_character'], config['avatar_style'])
logger.info('🗣️ Voice: %s', config['avatar_voice'])
