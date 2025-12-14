# voice_factory_backend

Backend FastAPI pour le projet Voice Factory : pipeline de traitement de la voix qui transforme des requêtes audio ou texte en réponses parlées avec visualisations.

## 🎯 Fonctionnalités

Le pipeline comprend :
1. **STT (Speech-to-Text)** : Transcription audio vers texte avec pyannote
2. **Text2SQL** : Génération de requêtes SQL depuis du langage naturel avec OpenAI
3. **Exécution SQL** : Requêtes sur des données CSV avec DuckDB
4. **Génération de réponse** : Création de réponses naturelles avec OpenAI
5. **Visualisation** : Génération de graphiques Plotly
6. **TTS (Text-to-Speech)** : Synthèse vocale avec Gradium

## 📋 Prérequis

- Python 3.12+
- Docker (optionnel, pour exécution en conteneur)
- Clés API :
  - `PYANNOTE_API_KEY` : API pyannote pour la transcription
  - `OPENAI_API_KEY` : API OpenAI pour Text2SQL et génération de réponses
  - `GRADIUM_API_KEY` : API Gradium pour la synthèse vocale

## 🚀 Installation

### Installation locale

1. Clonez le dépôt :
```bash
git clone <repository-url>
cd voice_factory_backend
```

2. Installez les dépendances avec `uv` :
```bash
pip install uv
uv sync
```

Ou avec `pip` :
```bash
pip install -e .
```

3. Configurez les variables d'environnement :
```bash
cp .env.example .env  # Si disponible
# Éditez .env et ajoutez vos clés API
```

### Installation avec Docker

```bash
docker build -t voice-factory-backend .
docker run -d -p 8000:8000 --env-file .env voice-factory-backend
```

## 🔧 Variables d'environnement

Variables requises :
- `PYANNOTE_API_KEY` : Clé API pyannote (requis pour STT)
- `OPENAI_API_KEY` : Clé API OpenAI (requis pour Text2SQL et génération de réponses)
- `GRADIUM_API_KEY` : Clé API Gradium (requis pour TTS)

Variables optionnelles :
- `LOG_LEVEL` : Niveau de log (CRITICAL, ERROR, WARNING, INFO, DEBUG). Défaut : `INFO`
- `VOICE_FACTORY_LOG_TEXT` : Afficher le contenu du texte dans les logs (`1`, `true`, `yes`, `y`, `on`). Défaut : `false`
- `CORS_ALLOW_ORIGINS` : Origines CORS autorisées, séparées par des virgules. Défaut : `http://localhost:3000,http://127.0.0.1:3000`

## 📖 Utilisation

### Mode CLI

Exécutez le pipeline complet depuis un fichier audio :

```bash
python main.py --audio path/to/audio.m4a --transcript-level turn
```

Options principales :
- `--audio` : Chemin vers le fichier audio (requis)
- `--transcript-level` : Niveau de transcription (`turn`, `word`, `both`). Défaut : `turn`
- `--text2sql-model` : Modèle OpenAI pour Text2SQL. Défaut : `gpt-5.2`
- `--output2answer-model` : Modèle OpenAI pour la génération de réponses. Défaut : `gpt-5.2`
- `--voice-id` : ID de voix Gradium. Défaut : `YTpq7expH9539ERJ`
- `--wav-out` : Fichier de sortie WAV. Défaut : `outputs/final_answer.wav`

### Mode API (FastAPI)

Démarrez le serveur :

```bash
uvicorn main_fastapi:app --host 0.0.0.0 --port 8000
```

Ou avec Docker :
```bash
docker run -p 8000:8000 --env-file .env voice-factory-backend
```

## 🌐 Endpoints API

### `GET /health`

Vérifie l'état du serveur.

**Réponse :**
```json
{"ok": true}
```

### `POST /v1/voice-factory/stt`

Endpoint STT uniquement : transcrit un fichier audio en texte.

**Requête (multipart/form-data) :**
- `audio` : Fichier audio (requis)
- `transcript_level` : `turn`, `word`, ou `both`. Défaut : `turn`

**Réponse :**
```json
{
  "question_text": "What's the temperature of machine one?",
  "transcript_level": "turn"
}
```

**Exemple avec curl :**
```bash
curl -X POST http://localhost:8000/v1/voice-factory/stt \
  -F "audio=@audio.m4a" \
  -F "transcript_level=turn"
```

### `POST /v1/voice-factory/answer`

Endpoint réponse uniquement : génère une réponse (Text2SQL + réponse + visualisation + TTS optionnel) depuis du texte.

**Requête (JSON) :**
```json
{
  "text": "What's the temperature of machine one?",
  "include_audio": true
}
```

**Réponse :**
```json
{
  "question_text": "What's the temperature of machine one?",
  "answer_text": "The temperature of machine one is 25.3°C.",
  "visualization": {
    "type": "plotly",
    "figure": {...}
  },
  "audio": {
    "filename": "answer.wav",
    "mime_type": "audio/wav",
    "audio_base64": "base64-encoded-audio-data"
  }
}
```

**Exemple avec curl :**
```bash
curl -X POST http://localhost:8000/v1/voice-factory/answer \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the temperature?", "include_audio": true}'
```

## 📁 Structure du projet

```
voice_factory_backend/
├── main.py              # Script CLI pour le pipeline complet
├── main_fastapi.py      # Application FastAPI
├── pyannote_stt.py      # Intégration STT avec pyannote
├── gradium_tts.py       # Intégration TTS avec Gradium
├── text2sql.py          # Génération SQL et visualisation
├── data/                # Données CSV (Machine_Data, Sensor_Data, Telemetry_Data)
├── prompts/             # Prompts pour OpenAI
│   ├── text2sql_prompt.txt
│   └── output2answer_prompt.txt
├── outputs/             # Fichiers générés (WAV, CSV, visualisations)
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 🗄️ Données

Le projet utilise trois fichiers CSV principaux :
- `Machine_Data.csv` : Données des machines
- `Sensor_Data.csv` : Données des capteurs
- `Telemetry_Data.csv` : Données de télémétrie

Ces fichiers sont chargés dans DuckDB pour l'exécution des requêtes SQL générées.

## 🔍 Documentation API interactive

Une fois le serveur démarré, accédez à :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **OpenAPI JSON** : http://localhost:8000/openapi.json

## 🐳 Docker

### Construction de l'image

```bash
docker build -t voice-factory-backend .
```

### Exécution

```bash
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name voice-factory-backend \
  voice-factory-backend
```

### Logs

```bash
docker logs -f voice-factory-backend
```

## 🧪 Tests

Pour tester l'API localement :

```bash
# Test health check
curl http://localhost:8000/health

# Test STT
curl -X POST http://localhost:8000/v1/voice-factory/stt \
  -F "audio=@test_query.m4a"

# Test answer
curl -X POST http://localhost:8000/v1/voice-factory/answer \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the temperature of machine one?", "include_audio": false}'
```

## 📝 Notes

- Les transcriptions sont nettoyées : les labels de locuteur (ex: `_00`) et les timestamps sont retirés pour ne garder que le texte.
- Les fichiers audio générés sont sauvegardés dans le dossier `outputs/`.
- Les visualisations sont générées au format Plotly JSON pour intégration frontend.

## 📄 Licence

[À compléter]
