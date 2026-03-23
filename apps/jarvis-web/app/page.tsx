"use client";

import { motion } from "framer-motion";
import { Mic, MicOff, PhoneCall, RefreshCw, Square, Volume2, VolumeX } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { ChatWindow } from "../components/ChatWindow";
import { RealtimeControls, type DetectionItem } from "../components/RealtimeControls";
import { SessionSidebar } from "../components/SessionSidebar";
import {
  getApiToken,
  interruptRealtime,
  openRealtimeSocket,
  pushCameraFrame,
  pushRealtimeSummary,
  sendRealtimeTurn,
  sendRealtimeWs,
  setApiToken,
  startRealtimeSession,
  startUrlStream
} from "../lib/api";
import type { ChatMessage, SessionItem } from "../lib/types";

function makeMsg(role: "user" | "assistant", content: string): ChatMessage {
  return {
    id: Math.random().toString(36).slice(2),
    role,
    content,
    ts: Date.now()
  };
}

export default function Page() {
  const Motion = motion as any;
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [activeId, setActiveId] = useState("");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [transport, setTransport] = useState<"ws" | "http">("http");
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [audioStatus, setAudioStatus] = useState("voice: en-IN");
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoiceUri, setSelectedVoiceUri] = useState("");
  const [showAllVoices, setShowAllVoices] = useState(false);
  const [micOn, setMicOn] = useState(false);
  const [liveDraftText, setLiveDraftText] = useState("");
  const [assistantDraftText, setAssistantDraftText] = useState("");
  const [apiToken, setApiTokenInput] = useState("");
  const [userId] = useState("web_user");
  const wsRef = useRef<WebSocket | null>(null);
  const wsSessionRef = useRef<string>("");
  const pendingRef = useRef<Set<string>>(new Set());
  const selectedVoiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const scriptRef = useRef<ScriptProcessorNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const liveDraftRef = useRef("");
  const assistantDraftRef = useRef("");
  const draftCommitTimerRef = useRef<number | null>(null);
  const speechRecRef = useRef<{
    start: () => void;
    stop: () => void;
    abort: () => void;
    onresult: ((event: any) => void) | null;
    onerror: ((event: any) => void) | null;
    onend: (() => void) | null;
    continuous: boolean;
    interimResults: boolean;
    lang: string;
  } | null>(null);
  const micDesiredRef = useRef(false);
  const vadSpeechStartedRef = useRef(false);
  const vadLastSpeechAtRef = useRef(0);
  const vadCommitLockRef = useRef(false);
  const assistantSpeakingRef = useRef(false);
  const lastAssistantResponseRef = useRef("");
  const sttSuspendedRef = useRef(false);
  const detectionSendAtRef = useRef(0);
  const detectionSigRef = useRef("");
  const VOICE_STORAGE_KEY = "jarvis_tts_voice_uri";

  const active = useMemo(() => sessions.find((s) => s.id === activeId), [sessions, activeId]);

  useEffect(() => {
    liveDraftRef.current = liveDraftText;
  }, [liveDraftText]);

  useEffect(() => {
    assistantDraftRef.current = assistantDraftText;
  }, [assistantDraftText]);

  useEffect(() => {
    setApiTokenInput(getApiToken());
  }, []);

  const displayedVoices = useMemo(() => {
    if (showAllVoices) return availableVoices;
    return availableVoices.filter((v) => v.lang.toLowerCase().startsWith("en"));
  }, [availableVoices, showAllVoices]);

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setAudioStatus("voice: unavailable");
      return;
    }
    const synth = window.speechSynthesis;
    const pickVoice = () => {
      const voices = synth.getVoices();
      setAvailableVoices(voices);
      const persisted = String(window.localStorage.getItem(VOICE_STORAGE_KEY) || "").trim();
      if (persisted) {
        const exact = voices.find((v) => (v.voiceURI || v.name) === persisted);
        if (exact) {
          selectedVoiceRef.current = exact;
          setSelectedVoiceUri(exact.voiceURI || exact.name);
          setAudioStatus("voice: " + exact.name);
          return;
        }
      }
      const byName = (patterns: RegExp[]) =>
        voices.find((v) => patterns.some((p) => p.test(v.name)));
      const inNatural =
        byName([/female/i, /neural/i, /samantha/i, /aria/i, /ava/i, /serena/i]) &&
        voices.find(
          (v) =>
            (/female/i.test(v.name) ||
              /neural/i.test(v.name) ||
              /samantha/i.test(v.name) ||
              /aria/i.test(v.name) ||
              /ava/i.test(v.name) ||
              /serena/i.test(v.name)) &&
            v.lang.toLowerCase().startsWith("en-in")
        );
      const inVoice =
        inNatural ||
        voices.find((v) => v.lang.toLowerCase() === "en-in") ||
        voices.find((v) => v.lang.toLowerCase().startsWith("en-in")) ||
        voices.find((v) => /india|indian/i.test(v.name));
      const enNatural = byName([/female/i, /neural/i, /samantha/i, /aria/i, /ava/i, /serena/i]);
      selectedVoiceRef.current =
        inVoice || enNatural || voices.find((v) => v.lang.toLowerCase().startsWith("en")) || null;
      setSelectedVoiceUri(selectedVoiceRef.current ? selectedVoiceRef.current.voiceURI || selectedVoiceRef.current.name : "");
      setAudioStatus(
        selectedVoiceRef.current
          ? "voice: " + selectedVoiceRef.current.name
          : "voice: en-IN (fallback)"
      );
    };
    pickVoice();
    synth.addEventListener("voiceschanged", pickVoice);
    return () => synth.removeEventListener("voiceschanged", pickVoice);
  }, []);

  function refreshVoiceList() {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    const voices = window.speechSynthesis.getVoices();
    setAvailableVoices(voices);
  }

  function updateSelectedVoice(uriOrName: string) {
    setSelectedVoiceUri(uriOrName);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(VOICE_STORAGE_KEY, uriOrName);
    }
    const voices = (availableVoices.length
      ? availableVoices
      : typeof window !== "undefined" && "speechSynthesis" in window
        ? window.speechSynthesis.getVoices()
        : []) as SpeechSynthesisVoice[];
    const selected =
      voices.find((v) => (v.voiceURI || v.name) === uriOrName) ||
      voices.find((v) => v.name === uriOrName) ||
      null;
    selectedVoiceRef.current = selected;
    setAudioStatus(selected ? "voice: " + selected.name : "voice: en-IN (fallback)");
  }

  function stopSpeaking() {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    assistantSpeakingRef.current = false;
    resumeMicAfterTts();
  }

  function suspendMicForTts() {
    sttSuspendedRef.current = true;
    if (draftCommitTimerRef.current !== null) {
      window.clearTimeout(draftCommitTimerRef.current);
      draftCommitTimerRef.current = null;
    }
    const rec = speechRecRef.current;
    if (rec) {
      try {
        rec.onend = null;
        rec.stop();
      } catch {
        // ignore
      }
      try {
        rec.abort();
      } catch {
        // ignore
      }
      speechRecRef.current = null;
    }
    const stream = micStreamRef.current;
    if (stream) {
      stream.getAudioTracks().forEach((t) => {
        t.enabled = false;
      });
    }
  }

  function resumeMicAfterTts() {
    if (!micDesiredRef.current) return;
    sttSuspendedRef.current = false;
    const stream = micStreamRef.current;
    if (stream) {
      stream.getAudioTracks().forEach((t) => {
        t.enabled = true;
      });
    }
    if (!speechRecRef.current) {
      startBrowserLiveTranscription();
    }
  }

  function speakAssistant(text: string) {
    if (!audioEnabled || !text.trim()) return;
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-IN";
    utterance.rate = 0.9;
    utterance.pitch = 0.92;
    utterance.volume = 0.88;
    utterance.onstart = () => {
      assistantSpeakingRef.current = true;
      suspendMicForTts();
    };
    utterance.onend = () => {
      assistantSpeakingRef.current = false;
      resumeMicAfterTts();
    };
    utterance.onerror = () => {
      assistantSpeakingRef.current = false;
      resumeMicAfterTts();
    };
    if (selectedVoiceRef.current) utterance.voice = selectedVoiceRef.current;
    synth.cancel();
    synth.speak(utterance);
  }

  function normalizeSpeechText(text: string): string {
    return String(text || "")
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function looksLikeAssistantEcho(candidate: string): boolean {
    const c = normalizeSpeechText(candidate);
    const a = normalizeSpeechText(lastAssistantResponseRef.current);
    if (!c || !a) return false;
    if (c === a) return true;
    if (c.length >= 18 && a.includes(c)) return true;
    if (a.length >= 18 && c.includes(a)) return true;
    return false;
  }

  function floatToPcm16B64(input: Float32Array): string {
    const out = new Int16Array(input.length);
    for (let i = 0; i < input.length; i += 1) {
      const s = Math.max(-1, Math.min(1, input[i]));
      out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
    }
    const bytes = new Uint8Array(out.buffer);
    let binary = "";
    const block = 0x8000;
    for (let i = 0; i < bytes.length; i += block) {
      binary += String.fromCharCode(...bytes.subarray(i, i + block));
    }
    return btoa(binary);
  }

  function calcRms(input: Float32Array): number {
    if (!input.length) return 0;
    let sum = 0;
    for (let i = 0; i < input.length; i += 1) {
      sum += input[i] * input[i];
    }
    return Math.sqrt(sum / input.length);
  }

  function startBrowserLiveTranscription() {
    if (typeof window === "undefined") return;
    const W = window as unknown as {
      SpeechRecognition?: new () => any;
      webkitSpeechRecognition?: new () => any;
    };
    const Ctor = W.SpeechRecognition || W.webkitSpeechRecognition;
    if (!Ctor) return;
    const rec = new Ctor();
    rec.lang = "en-IN";
    rec.continuous = true;
    rec.interimResults = true;
    rec.onresult = (event: any) => {
      if (assistantSpeakingRef.current) {
        setLiveDraftText("");
        return;
      }
      let interim = "";
      let finalCombined = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const segment = String(event.results[i]?.[0]?.transcript || "");
        if (!segment) continue;
        if (event.results[i].isFinal) {
          finalCombined += segment.trim() + " ";
        } else {
          interim += segment + " ";
        }
      }
      setLiveDraftText(interim.trim());
      if (draftCommitTimerRef.current !== null) {
        window.clearTimeout(draftCommitTimerRef.current);
        draftCommitTimerRef.current = null;
      }
      const finalText = finalCombined.trim();
      if (finalText) {
        setLiveDraftText("");
        void submitVoiceTurn(finalText);
        return;
      }
      const maybeDraft = interim.trim();
      if (maybeDraft) {
        draftCommitTimerRef.current = window.setTimeout(() => {
          const text = liveDraftRef.current.trim();
          if (!text || !micDesiredRef.current) return;
          setLiveDraftText("");
          void submitVoiceTurn(text);
        }, 1150);
      }
    };
    rec.onerror = () => {
      // Keep backend STT as primary path; browser interim is optional UX.
    };
    rec.onend = () => {
      if (sttSuspendedRef.current) {
        return;
      }
      const pending = liveDraftRef.current.trim();
      if (pending && micDesiredRef.current) {
        setLiveDraftText("");
        void submitVoiceTurn(pending);
      }
      if (micDesiredRef.current) {
        try {
          rec.start();
        } catch {
          // ignore
        }
      }
    };
    speechRecRef.current = rec;
    try {
      rec.start();
    } catch {
      // ignore
    }
  }

  async function submitVoiceTurn(text: string) {
    if (!active || !text.trim()) return;
    const clean = text.trim();
    if (assistantSpeakingRef.current || looksLikeAssistantEcho(clean)) {
      setLiveDraftText("");
      return;
    }
    appendMessage(active.id, makeMsg("user", clean));
    const cmdId = Math.random().toString(36).slice(2);
    const wsSent = sendRealtimeWs(wsRef.current, {
      type: "turn",
      id: cmdId,
      text: clean,
      modality: "voice",
      context: { realtime_mode: true, source: "browser_live_stt" }
    });
    if (wsSent) {
      pendingRef.current.add(cmdId);
      setBusy(true);
      setTransport("ws");
      return;
    }
    setTransport("http");
    setBusy(true);
    try {
      const out = await sendRealtimeTurn(active.id, clean);
      appendMessage(active.id, makeMsg("assistant", out || "No response"));
      lastAssistantResponseRef.current = out || "No response";
      speakAssistant(out || "No response");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Turn failed");
    } finally {
      setBusy(false);
    }
  }

  function stopBrowserLiveTranscription() {
    setLiveDraftText("");
    liveDraftRef.current = "";
    if (draftCommitTimerRef.current !== null) {
      window.clearTimeout(draftCommitTimerRef.current);
      draftCommitTimerRef.current = null;
    }
    micDesiredRef.current = false;
    const rec = speechRecRef.current;
    if (!rec) return;
    try {
      rec.onend = null;
      rec.stop();
    } catch {
      // ignore
    }
    try {
      rec.abort();
    } catch {
      // ignore
    }
    speechRecRef.current = null;
  }

  async function startMic() {
    if (micOn || !active) return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError("Microphone streaming requires active WebSocket transport.");
      return;
    }
    try {
      setError("");
      micDesiredRef.current = true;
      startBrowserLiveTranscription();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        },
        video: false
      });
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) {
        throw new Error("Web Audio API is unavailable in this browser.");
      }
      const ctx = new AudioCtx();
      const source = ctx.createMediaStreamSource(stream);
      const script = ctx.createScriptProcessor(4096, 1, 1);
      const vadThreshold = 0.015;
      const vadSilenceMs = 900;
      script.onaudioprocess = (ev) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        // When browser live STT is active, it owns transcript commit + turn sending.
        // Keep backend audio STT path as fallback only when browser STT is unavailable.
        if (speechRecRef.current) return;
        const ch = ev.inputBuffer.getChannelData(0);
        const now = Date.now();
        const rms = calcRms(ch);
        const speakingNow = rms >= vadThreshold;

        if (speakingNow) {
          if (!vadSpeechStartedRef.current) {
            // Barge-in: as soon as user starts speaking, stop assistant speech.
            stopSpeaking();
            sendRealtimeWs(wsRef.current, { type: "interrupt", reason: "barge_in_voice" });
          }
          vadSpeechStartedRef.current = true;
          vadLastSpeechAtRef.current = now;
        }

        const pcm16_b64 = floatToPcm16B64(ch);
        sendRealtimeWs(wsRef.current, {
          type: "audio_chunk",
          pcm16_b64,
          sample_rate: ctx.sampleRate
        });

        if (
          vadSpeechStartedRef.current &&
          !speakingNow &&
          now - vadLastSpeechAtRef.current >= vadSilenceMs &&
          !vadCommitLockRef.current
        ) {
          vadCommitLockRef.current = true;
          sendRealtimeWs(wsRef.current, {
            type: "audio_commit",
            id: Math.random().toString(36).slice(2),
            language: "en-IN",
            auto_turn: true
          });
          vadSpeechStartedRef.current = false;
          vadLastSpeechAtRef.current = 0;
          window.setTimeout(() => {
            vadCommitLockRef.current = false;
          }, 320);
        }
      };
      source.connect(script);
      script.connect(ctx.destination);
      audioCtxRef.current = ctx;
      audioSourceRef.current = source;
      scriptRef.current = script;
      micStreamRef.current = stream;
      setMicOn(true);
    } catch (err) {
      setMicOn(false);
      setError(err instanceof Error ? err.message : "Failed to start microphone");
    }
  }

  async function stopMic(commit = true) {
    stopBrowserLiveTranscription();
    if (scriptRef.current) {
      scriptRef.current.disconnect();
      scriptRef.current.onaudioprocess = null;
      scriptRef.current = null;
    }
    if (audioSourceRef.current) {
      audioSourceRef.current.disconnect();
      audioSourceRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      await audioCtxRef.current.close().catch(() => undefined);
      audioCtxRef.current = null;
    }
    if (commit && vadSpeechStartedRef.current && !speechRecRef.current) {
      sendRealtimeWs(wsRef.current, {
        type: "audio_commit",
        id: Math.random().toString(36).slice(2),
        language: "en-IN",
        auto_turn: true
      });
    }
    vadSpeechStartedRef.current = false;
    vadLastSpeechAtRef.current = 0;
    vadCommitLockRef.current = false;
    setMicOn(false);
  }

  useEffect(() => {
    if (!activeId) return;
    if (wsRef.current && wsSessionRef.current === activeId && wsRef.current.readyState <= 1) return;
    void stopMic(false);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      wsSessionRef.current = "";
      pendingRef.current.clear();
      setBusy(false);
    }
    const ws = openRealtimeSocket(activeId, {
      onOpen: () => setTransport("ws"),
      onClose: () => {
        void stopMic(false);
        setTransport("http");
        pendingRef.current.clear();
        setBusy(false);
      },
      onError: () => setTransport("http"),
      onEvent: (event) => {
        if (event.type === "progress") {
          const stage = String(event.stage || "processing").trim();
          const msg = String(event.message || "Processing...").trim();
          const elapsedMsRaw = Number(event.elapsed_ms || 0);
          const elapsedSec = Number.isFinite(elapsedMsRaw) ? Math.max(0, elapsedMsRaw / 1000) : 0;
          const pretty = elapsedSec > 0 ? `${msg} (${elapsedSec.toFixed(1)}s)` : msg;
          setAssistantDraftText(`[${stage}] ${pretty}`);
          return;
        }
        if (event.type === "response") {
          const response = String(event.response || "").trim() || "No response";
          setAssistantDraftText("");
          appendMessage(activeId, makeMsg("assistant", response));
          lastAssistantResponseRef.current = response;
          speakAssistant(response);
          const id = String(event.id || "");
          if (id) pendingRef.current.delete(id);
          setBusy(pendingRef.current.size > 0);
          return;
        }
        if (event.type === "error") {
          const id = String(event.id || "");
          if (id) pendingRef.current.delete(id);
          setBusy(pendingRef.current.size > 0);
          setAssistantDraftText("");
          setError(String(event.error || "Realtime request failed"));
          return;
        }
        if (event.type === "response_started") {
          setAssistantDraftText("");
          return;
        }
        if (event.type === "response_section") {
          const sec = String(event.section_text || "").trim();
          if (!sec) return;
          setAssistantDraftText((prev) => (prev ? prev + "\n\n" + sec : sec));
          return;
        }
        if (event.type === "response_done") {
          const response = String(event.response || "").trim();
          const finalText = response || assistantDraftRef.current || "No response";
          setAssistantDraftText("");
          appendMessage(activeId, makeMsg("assistant", finalText));
          lastAssistantResponseRef.current = finalText;
          speakAssistant(finalText);
          const id = String(event.id || "");
          if (id) pendingRef.current.delete(id);
          setBusy(pendingRef.current.size > 0);
          return;
        }
        if (event.type === "interrupt_ack") {
          // Interrupt acknowledgements are transport/control signals; keep chat history clean.
          return;
        }
        if (event.type === "transcript") {
          const text = String(event.text || "").trim();
          if (text) {
            setLiveDraftText("");
            appendMessage(activeId, makeMsg("user", text));
          }
        }
      }
    });
    wsRef.current = ws;
    wsSessionRef.current = activeId;
    return () => {
      if (wsRef.current && wsSessionRef.current === activeId) {
        wsRef.current.close();
        wsRef.current = null;
        wsSessionRef.current = "";
        pendingRef.current.clear();
        setBusy(false);
      }
    };
  }, [activeId]);

  useEffect(() => {
    return () => {
      void stopMic(false);
      stopBrowserLiveTranscription();
      stopSpeaking();
    };
  }, []);

  async function createSession() {
    setError("");
    try {
      const sid = await startRealtimeSession(userId);
      const item: SessionItem = {
        id: sid,
        title: "Session " + (sessions.length + 1),
        createdAt: Date.now(),
        messages: [makeMsg("assistant", "Realtime session started. You can chat, interrupt, and stream camera.")]
      };
      setSessions((prev) => [item, ...prev]);
      setActiveId(sid);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session");
    }
  }

  function appendMessage(sessionId: string, msg: ChatMessage) {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sessionId) return s;
        return { ...s, messages: [...s.messages, msg] };
      })
    );
  }

  async function sendTurn() {
    if (!active || !input.trim() || busy) return;
    setError("");
    const text = input.trim();
    setInput("");
    appendMessage(active.id, makeMsg("user", text));
    const cmdId = Math.random().toString(36).slice(2);
    const wsSent = sendRealtimeWs(wsRef.current, {
      type: "turn",
      id: cmdId,
      text,
      modality: "voice",
      context: { realtime_mode: true }
    });
    if (wsSent) {
      pendingRef.current.add(cmdId);
      setBusy(true);
      setTransport("ws");
      return;
    }
    setTransport("http");
    setBusy(true);
    try {
      const out = await sendRealtimeTurn(active.id, text);
      appendMessage(active.id, makeMsg("assistant", out || "No response"));
      speakAssistant(out || "No response");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Turn failed");
    } finally {
      setBusy(false);
    }
  }

  async function onInterrupt() {
    if (!active) return;
    stopSpeaking();
    const wsSent = sendRealtimeWs(wsRef.current, { type: "interrupt", reason: "ui_interrupt" });
    if (wsSent) return;
    await interruptRealtime(active.id, "ui_interrupt");
  }

  async function onPushFrame(imageB64: string) {
    if (!active) return;
    const wsSent = sendRealtimeWs(wsRef.current, {
      type: "media",
      source: "iphone_camera",
      image_b64: imageB64,
      metadata: { source_device: "webcam" }
    });
    if (wsSent) return;
    await pushCameraFrame(active.id, imageB64, "iphone_camera");
  }

  async function onDetections(detections: DetectionItem[]) {
    if (!active || detections.length === 0) return;
    const top = detections.slice(0, 5);
    const signature = top
      .map((d) => `${d.label}:${Math.round(d.score * 100)}`)
      .join("|");
    const now = Date.now();
    if (signature === detectionSigRef.current && now - detectionSendAtRef.current < 4500) return;
    if (now - detectionSendAtRef.current < 1200) return;

    detectionSigRef.current = signature;
    detectionSendAtRef.current = now;
    const summary =
      "Detected in camera feed: " +
      top.map((d) => `${d.label} (${Math.round(d.score * 100)}%)`).join(", ") +
      ".";
    const metadata = {
      source_device: "webcam",
      detections: top
    };
    const wsSent = sendRealtimeWs(wsRef.current, {
      type: "media",
      source: "camera_detection",
      summary,
      metadata
    });
    if (wsSent) return;
    await pushRealtimeSummary(active.id, summary, "camera_detection", metadata);
  }

  async function onStartUrl(sourceUrl: string, sourceType: "rtsp" | "webrtc" | "http") {
    if (!active) return;
    const wsSent = sendRealtimeWs(wsRef.current, {
      type: "start_stream",
      source_url: sourceUrl,
      source_type: sourceType,
      interval_ms: 1500
    });
    if (wsSent) return;
    await startUrlStream(active.id, sourceUrl, sourceType);
  }

  return (
    <main className="appShell">
      <SessionSidebar sessions={sessions} activeId={activeId} onCreate={createSession} onSelect={setActiveId} />
      <section className="mainColumn">
        <Motion.div
          className="hero glass"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          <div className="heroTitleRow">
            <h1>Jarvis Live Console</h1>
            <span className="heroPill">{transport.toUpperCase()}</span>
          </div>
          <p>Multi-session voice conversation with realtime visual grounding</p>
          <div className="heroControls">
            <input
              className="tokenInput"
              type="password"
              value={apiToken}
              onChange={(e) => {
                const token = e.target.value;
                setApiTokenInput(token);
                setApiToken(token);
              }}
              placeholder="JARVIS API token (Bearer)"
            />
            <button className="btn" onClick={() => setAudioEnabled((v) => !v)}>
              {audioEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
              {audioEnabled ? "Mute Voice" : "Enable Voice"}
            </button>
            <select
              className="voiceSelect"
              value={selectedVoiceUri}
              onChange={(e) => updateSelectedVoice(e.target.value)}
            >
              {displayedVoices.length === 0 ? <option value="">Default Voice</option> : null}
              {displayedVoices.map((v) => (
                <option key={(v.voiceURI || v.name) + v.lang} value={v.voiceURI || v.name}>
                  {v.name} ({v.lang})
                </option>
              ))}
            </select>
            <button className="btn" onClick={refreshVoiceList}>
              <RefreshCw size={16} />
              Refresh Voices
            </button>
            <button className="btn" onClick={() => setShowAllVoices((v) => !v)}>
              {showAllVoices ? "English Voices" : "All Voices"}
            </button>
            <button
              className="btn"
              disabled={!active || transport !== "ws"}
              onClick={() => {
                void (micOn ? stopMic(true) : startMic());
              }}
            >
              {micOn ? <MicOff size={16} /> : <PhoneCall size={16} />}
              {micOn ? "End Call" : "Start Call"}
            </button>
            <button className="btn" onClick={stopSpeaking}>
              <Square size={16} />
              Stop Speaking
            </button>
            <span className="voiceBadge">{audioStatus}</span>
            <span className={"voiceBadge" + (micOn ? " live" : "")}>
              {micOn ? <Mic size={12} /> : <MicOff size={12} />}
              {micOn ? "call: live" : "call: off"}
            </span>
          </div>
        </Motion.div>

        {active ? (
          <>
            <ChatWindow
              messages={active.messages}
              live={micOn}
              liveDraftText={liveDraftText}
              assistantDraftText={assistantDraftText}
            />
            <RealtimeControls
              connected={!!active}
              onInterrupt={onInterrupt}
              onPushFrame={onPushFrame}
              onStartUrlStream={onStartUrl}
              onDetections={onDetections}
            />
            <div className="composer glass">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Talk to Jarvis..."
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void sendTurn();
                  }
                }}
              />
              <button className="btn" disabled={busy} onClick={sendTurn}>
                {busy ? "Sending..." : "Send"}
              </button>
            </div>
          </>
        ) : (
          <section className="emptyState glass">
            <h3>Create your first realtime session</h3>
            <p>Open multiple sessions, stream from camera, and converse continuously.</p>
            <button className="btn" onClick={createSession}>
              Start Session
            </button>
          </section>
        )}
        {error ? <p className="hint error">{error}</p> : null}
      </section>
    </main>
  );
}
