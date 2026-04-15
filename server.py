import os
import tempfile
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from app import (
    config,
    get_acs_token,
    get_speech_token,
    get_avatar_ice_config,
    transcribe_audio_file,
    get_agent_answer,
    process_question,
    foundry_error,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='public', static_url_path='')
app.config['JSON_SORT_KEYS'] = False
CORS(app)


def require_api_key(req):
    configured_key = os.getenv('APP_API_KEY')
    if not configured_key:
        return True
    provided_key = req.headers.get('x-app-api-key')
    return provided_key == configured_key


def cleanup_temp_file(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


@app.route('/health', methods=['GET'])
def health():
    name, version = None, None
    try:
        from app import get_agent_reference
        name, version = get_agent_reference()
    except Exception:
        pass

    return jsonify({
        'status': 'running',
        'foundry_endpoint': config['foundry_endpoint'] or None,
        'agent_reference': config['agent_reference'] or None,
        'agent_name': name,
        'agent_version': version,
        'base_model': config['foundry_base_model'],
        'whisper_deployment': config['whisper_deployment'],
        'foundry_error': foundry_error,
    })


@app.route('/acs-token', methods=['GET'])
def acs_token():
    if not require_api_key(request):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        token = get_acs_token()
        return jsonify(token)
    except Exception as ex:
        logger.exception('ACS token error')
        return jsonify({'error': str(ex)}), 500


@app.route('/speech-token', methods=['GET'])
def speech_token():
    if not require_api_key(request):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        token = get_speech_token()
        return jsonify(token)
    except Exception as ex:
        logger.exception('Speech token error')
        return jsonify({'error': str(ex)}), 500


@app.route('/avatar-ice', methods=['GET'])
def avatar_ice():
    if not require_api_key(request):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        config_data = get_avatar_ice_config()
        return jsonify(config_data)
    except Exception as ex:
        logger.exception('Avatar ICE error')
        return jsonify({'error': str(ex)}), 500


@app.route('/process-audio', methods=['POST'])
def process_audio():
    if not require_api_key(request):
        return jsonify({'error': 'Unauthorized'}), 401
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio = request.files['audio']
    suffix = os.path.splitext(audio.filename)[1] or '.webm'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio.read())
        temp_path = tmp.name

    try:
        result = process_question(open(temp_path, 'rb').read())
        return jsonify({'success': True, 'result': result})
    except Exception as ex:
        logger.exception('Process audio error')
        return jsonify({'error': str(ex)}), 500
    finally:
        cleanup_temp_file(temp_path)


@app.route('/ask', methods=['POST'])
def ask():
    if not require_api_key(request):
        return jsonify({'error': 'Unauthorized'}), 401

    question = (request.json or {}).get('question', '')
    if not isinstance(question, str) or not question.strip():
        return jsonify({'error': 'No question provided'}), 400

    try:
        answer = get_agent_answer(question.strip())
        return jsonify({'question': question.strip(), 'answer': answer})
    except Exception as ex:
        logger.exception('Ask error')
        return jsonify({'error': str(ex)}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not require_api_key(request):
        return jsonify({'error': 'Unauthorized'}), 401
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio = request.files['audio']
    suffix = os.path.splitext(audio.filename)[1] or '.webm'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio.read())
        temp_path = tmp.name

    try:
        text = transcribe_audio_file(temp_path)
        return jsonify({'text': text})
    except Exception as ex:
        logger.exception('Transcribe error')
        return jsonify({'error': str(ex)}), 500
    finally:
        cleanup_temp_file(temp_path)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return app.send_static_file(path)
    return app.send_static_file('index.html')


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5015))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', '0') == '1')
