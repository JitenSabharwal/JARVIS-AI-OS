# Jarvis Web Console (Next.js)

Realtime, multi-session conversation UI for Jarvis.

## Run

```bash
cd apps/jarvis-web
npm install
NEXT_PUBLIC_JARVIS_API_BASE=http://127.0.0.1:8080 \
NEXT_PUBLIC_JARVIS_API_TOKEN=<token> \
npm run dev
```

Open `http://127.0.0.1:3001`.

## Notes

- The app uses WebSocket duplex transport first:
  - `GET /api/v1/realtime/sessions/{session_id}/ws?access_token=<token>`
  - Falls back to HTTP realtime endpoints automatically if WS is unavailable.
- Assistant replies are spoken in browser using TTS (`en-IN` default voice/lang) with mute/stop controls.
- Voice-in (microphone) streams PCM16 chunks over WebSocket (`audio_chunk`) and commits (`audio_commit`) for STT + auto-turn.
- Call mode is continuous: end-of-utterance is detected automatically with client-side VAD (silence-based), so no manual stop per sentence.
- UI shows live speech draft text while speaking, then keeps finalized transcript messages in chat history.
- Current backend STT uses local MLX Whisper (`mlx_whisper.transcribe`) by default (no cloud STT dependency).
- For iPhone camera streaming, open the site via HTTPS on iOS Safari and use **Start Camera Stream**.
- Camera mode sends periodic `image_b64` frames to:
  - `POST /api/v1/realtime/sessions/{session_id}/media`
- Live object detection overlay:
  - Runs locally in-browser using COCO-SSD (`@tensorflow-models/coco-ssd` + `@tensorflow/tfjs`).
  - Draws bounding boxes + labels directly on the video feed.
  - Sends compact detection summaries to realtime context (`source=camera_detection`) so Jarvis can reference visible items.
- URL stream mode starts background ingest via:
  - `POST /api/v1/realtime/sessions/{session_id}/streams/start`
