"use client";

import { motion } from "framer-motion";
import { Camera, CameraOff, Link2, Radio, ShieldAlert } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  deleteVisionIdentity,
  enrollVisionIdentity,
  listVisionIdentities,
  recognizeVisionIdentities,
  type VisionIdentity
} from "../lib/api";

export type DetectionItem = {
  label: string;
  score: number;
  bbox: [number, number, number, number];
  trackId?: string;
  identity?: string;
  identityScore?: number;
  personId?: string;
};

type PersonTrack = {
  bbox: [number, number, number, number];
  identity?: string;
  identityScore?: number;
  personId?: string;
  lastSeenAt: number;
};

type CoverageStats = {
  tracked: number;
  matched: number;
  scanning: number;
};

const DETECTION_COLORS = [
  "#22c55e", // green
  "#ef4444", // red
  "#f59e0b", // orange
  "#3b82f6", // blue
  "#eab308", // yellow
  "#06b6d4", // cyan
  "#a855f7", // purple
  "#ec4899" // pink
];

function colorForLabel(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) {
    hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  }
  return DETECTION_COLORS[hash % DETECTION_COLORS.length] || DETECTION_COLORS[0];
}

function bboxIou(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ax2 = a[0] + a[2];
  const ay2 = a[1] + a[3];
  const bx2 = b[0] + b[2];
  const by2 = b[1] + b[3];
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a[2]) * Math.max(0, a[3]);
  const areaB = Math.max(0, b[2]) * Math.max(0, b[3]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

type Props = {
  connected: boolean;
  onInterrupt: () => Promise<void>;
  onStartUrlStream: (sourceUrl: string, sourceType: "rtsp" | "webrtc" | "http") => Promise<void>;
  onPushFrame: (imageB64: string) => Promise<void>;
  onDetections?: (detections: DetectionItem[]) => Promise<void> | void;
};

export function RealtimeControls({
  connected,
  onInterrupt,
  onStartUrlStream,
  onPushFrame,
  onDetections
}: Props) {
  const Motion = motion as any;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const pushTimerRef = useRef<number | null>(null);
  const detectTimerRef = useRef<number | null>(null);
  const detectBusyRef = useRef(false);
  const cameraOnRef = useRef(false);
  const modelRef = useRef<any>(null);
  const detectionPushAtRef = useRef(0);
  const recognitionBusyRef = useRef(false);
  const recognitionAtRef = useRef(0);
  const personTracksRef = useRef<Map<string, PersonTrack>>(new Map());
  const trackSeqRef = useRef(0);
  const [cameraOn, setCameraOn] = useState(false);
  const [busy, setBusy] = useState(false);
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceType, setSourceType] = useState<"rtsp" | "webrtc" | "http">("http");
  const [cameraError, setCameraError] = useState("");
  const [detectorStatus, setDetectorStatus] = useState("detector: idle");
  const [detections, setDetections] = useState<DetectionItem[]>([]);
  const [detectionEnabled, setDetectionEnabled] = useState(true);
  const detectionEnabledRef = useRef(true);
  const [recognitionEnabled, setRecognitionEnabled] = useState(true);
  const [identityName, setIdentityName] = useState("");
  const [identityBusy, setIdentityBusy] = useState(false);
  const [identityStatus, setIdentityStatus] = useState("identity: idle");
  const [identities, setIdentities] = useState<VisionIdentity[]>([]);
  const [coverage, setCoverage] = useState<CoverageStats>({ tracked: 0, matched: 0, scanning: 0 });

  useEffect(() => {
    return () => {
      if (pushTimerRef.current !== null) {
        window.clearInterval(pushTimerRef.current);
      }
      if (detectTimerRef.current !== null) {
        window.clearInterval(detectTimerRef.current);
      }
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  useEffect(() => {
    detectionEnabledRef.current = detectionEnabled;
  }, [detectionEnabled]);

  useEffect(() => {
    if (!recognitionEnabled) personTracksRef.current.clear();
    if (!recognitionEnabled) setCoverage((prev) => ({ ...prev, matched: 0, scanning: prev.tracked }));
  }, [recognitionEnabled]);

  useEffect(() => {
    if (!detectionEnabled) setCoverage({ tracked: 0, matched: 0, scanning: 0 });
  }, [detectionEnabled]);

  useEffect(() => {
    void refreshIdentities();
  }, []);

  async function refreshIdentities() {
    try {
      const rows = await listVisionIdentities();
      setIdentities(rows);
      setIdentityStatus(`identity: ${rows.length} enrolled`);
    } catch (err) {
      setIdentityStatus(err instanceof Error ? "identity: " + err.message : "identity: unavailable");
    }
  }

  function clamp(val: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, val));
  }

  function cropDetectionAsJpeg(video: HTMLVideoElement, bbox: [number, number, number, number]): string {
    const [x, y, w, h] = bbox;
    const vw = video.videoWidth || 0;
    const vh = video.videoHeight || 0;
    if (!vw || !vh) return "";
    const sx = clamp(Math.floor(x), 0, Math.max(0, vw - 1));
    const sy = clamp(Math.floor(y), 0, Math.max(0, vh - 1));
    const sw = clamp(Math.floor(w), 20, Math.max(20, vw - sx));
    const sh = clamp(Math.floor(h), 20, Math.max(20, vh - sy));
    const canvas = document.createElement("canvas");
    canvas.width = 160;
    canvas.height = 160;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.75);
  }

  const canUseMedia = useMemo(
    () => typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia,
    []
  );

  async function ensureDetector() {
    if (modelRef.current) return modelRef.current;
    setDetectorStatus("detector: loading");
    const tf = await import("@tensorflow/tfjs");
    const cocoSsd = await import("@tensorflow-models/coco-ssd");
    await tf.ready();
    modelRef.current = await cocoSsd.load({ base: "lite_mobilenet_v2" });
    setDetectorStatus("detector: ready");
    return modelRef.current;
  }

  function drawDetections(items: DetectionItem[]) {
    if (!videoRef.current || !overlayRef.current) return;
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const width = video.videoWidth || 960;
    const height = video.videoHeight || 540;
    overlay.width = width;
    overlay.height = height;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 3;
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    for (const item of items) {
      const [x, y, w, h] = item.bbox;
      const scorePct = Math.round(item.score * 100);
      const idPct = typeof item.identityScore === "number" ? Math.round(item.identityScore * 100) : 0;
      const title = item.identity
        ? `${item.identity} (${item.label}) ${scorePct}% id:${idPct}%`
        : `${item.label} ${scorePct}%`;
      const color = colorForLabel(item.identity || item.label);
      ctx.strokeStyle = color;
      ctx.fillStyle = `${color}22`;
      ctx.strokeRect(x, y, w, h);
      // Light tint helps readability without hiding the video feed.
      ctx.fillRect(x, y, w, h);
      const textW = ctx.measureText(title).width + 10;
      const textY = Math.max(16, y - 4);
      ctx.fillStyle = "rgba(4, 15, 20, 0.88)";
      ctx.fillRect(x, textY - 14, textW, 16);
      ctx.fillStyle = color;
      ctx.fillText(title, x + 5, textY - 2);
    }
  }

  async function runDetectionTick() {
    if (!detectionEnabledRef.current || detectBusyRef.current || !videoRef.current || !cameraOnRef.current) return;
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight) return;
    detectBusyRef.current = true;
    try {
      const model = await ensureDetector();
      const raw = await model.detect(video, 20, 0.25);
      const mapped: DetectionItem[] = (Array.isArray(raw) ? raw : [])
        .filter((p: any) => Array.isArray(p?.bbox) && p.bbox.length === 4)
        .map((p: any) => ({
          label: String(p.class || "object"),
          score: Number(p.score || 0),
          bbox: [
            Number(p.bbox[0] || 0),
            Number(p.bbox[1] || 0),
            Number(p.bbox[2] || 0),
            Number(p.bbox[3] || 0)
          ] as [number, number, number, number]
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 10);
      let resolved = mapped;
      const now = Date.now();
      const tracks = personTracksRef.current;
      for (const [key, track] of tracks.entries()) {
        if (now - track.lastSeenAt > 10000) tracks.delete(key);
      }
      const personWithIdx = mapped
        .map((d, idx) => ({ d, idx }))
        .filter((x) => x.d.label === "person");
      const usedTrackIds = new Set<string>();
      const trackByDetectionIdx = new Map<number, string>();
      for (const { d, idx } of personWithIdx) {
        let best: { key: string; iou: number } | null = null;
        for (const [key, track] of tracks.entries()) {
          if (usedTrackIds.has(key)) continue;
          const iou = bboxIou(d.bbox, track.bbox);
          if (iou < 0.45) continue;
          if (!best || iou > best.iou) best = { key, iou };
        }
        const trackId = best ? best.key : `trk-${++trackSeqRef.current}`;
        const previous = tracks.get(trackId);
        tracks.set(trackId, {
          bbox: d.bbox,
          identity: previous?.identity,
          identityScore: previous?.identityScore,
          personId: previous?.personId,
          lastSeenAt: now
        });
        usedTrackIds.add(trackId);
        trackByDetectionIdx.set(idx, trackId);
        if (previous?.identity && previous?.personId) {
          resolved[idx] = {
            ...resolved[idx],
            trackId,
            identity: previous.identity,
            identityScore: previous.identityScore,
            personId: previous.personId
          };
        } else {
          resolved[idx] = {
            ...resolved[idx],
            trackId
          };
        }
      }
      const toRecognize = personWithIdx
        .filter(({ idx }) => {
          const trackId = trackByDetectionIdx.get(idx);
          if (!trackId) return false;
          const track = tracks.get(trackId);
          return !track?.personId;
        })
        .sort((a, b) => b.d.bbox[2] * b.d.bbox[3] - a.d.bbox[2] * a.d.bbox[3])
        .slice(0, 2);
      if (recognitionEnabled && toRecognize.length > 0 && !recognitionBusyRef.current) {
        if (now - recognitionAtRef.current > 900) {
          recognitionBusyRef.current = true;
          recognitionAtRef.current = now;
          try {
            const samples = toRecognize
              .map(({ d, idx }) => ({
                sample_id: "det-" + idx,
                detection_index: idx,
                image_b64: cropDetectionAsJpeg(video, d.bbox)
              }))
              .filter((s) => !!s.image_b64);
            if (samples.length > 0) {
              const matches = await recognizeVisionIdentities(samples);
              const byIdx = new Map<number, { name: string; score: number; personId: string }>();
              for (const m of matches) {
                const idx = Number(m.detection_index);
                const name = String(m.display_name || m.candidate_display_name || "").trim();
                const personId = String(m.person_id || m.candidate_person_id || "").trim();
                const score = Number(m.score || 0);
                const allowUnknownCandidate = Boolean(m.unknown) && score >= 0.68;
                if (m.unknown && !allowUnknownCandidate) continue;
                if (!Number.isFinite(idx) || !name || !personId) continue;
                byIdx.set(idx, { name, score, personId });
              }
              resolved = mapped.map((item, idx) => {
                const found = byIdx.get(idx);
                if (!found) return item;
                const trackId = trackByDetectionIdx.get(idx);
                if (trackId) {
                  tracks.set(trackId, {
                    bbox: item.bbox,
                    identity: found.name,
                    identityScore: found.score,
                    personId: found.personId,
                    lastSeenAt: now
                  });
                }
                return {
                  ...item,
                  trackId: trackByDetectionIdx.get(idx),
                  identity: found.name,
                  identityScore: found.score,
                  personId: found.personId
                };
              });
            }
          } catch {
            // Keep detection loop resilient if recognition endpoint is unavailable.
          } finally {
            recognitionBusyRef.current = false;
          }
        }
      }
      const visibleMatched = Array.from(usedTrackIds).reduce((acc, trackId) => {
        const t = tracks.get(trackId);
        return t?.personId ? acc + 1 : acc;
      }, 0);
      const visibleTracked = usedTrackIds.size;
      setCoverage({
        tracked: visibleTracked,
        matched: visibleMatched,
        scanning: Math.max(0, visibleTracked - visibleMatched)
      });

      setDetections(resolved);
      drawDetections(resolved);
      if (onDetections && Date.now() - detectionPushAtRef.current > 1800) {
        detectionPushAtRef.current = Date.now();
        await onDetections(resolved);
      }
    } catch (err) {
      setDetectorStatus(err instanceof Error ? "detector: " + err.message : "detector: unavailable");
    } finally {
      detectBusyRef.current = false;
    }
  }

  async function toggleCamera() {
    if (!connected) return;
    if (cameraOn) {
      if (pushTimerRef.current !== null) {
        window.clearInterval(pushTimerRef.current);
        pushTimerRef.current = null;
      }
      if (detectTimerRef.current !== null) {
        window.clearInterval(detectTimerRef.current);
        detectTimerRef.current = null;
      }
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      setDetections([]);
      personTracksRef.current.clear();
      trackSeqRef.current = 0;
      setCoverage({ tracked: 0, matched: 0, scanning: 0 });
      drawDetections([]);
      cameraOnRef.current = false;
      setCameraOn(false);
      setDetectorStatus("detector: idle");
      return;
    }
    if (!canUseMedia) {
      setCameraError("Camera capture not supported in this browser.");
      return;
    }
    try {
      setCameraError("");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 960, height: 540 },
        audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      cameraOnRef.current = true;
      setCameraOn(true);
      pushTimerRef.current = window.setInterval(async () => {
        if (!videoRef.current || !canvasRef.current) return;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth || 960;
        canvas.height = video.videoHeight || 540;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.55);
        await onPushFrame(dataUrl);
      }, 2200);
      detectTimerRef.current = window.setInterval(() => {
        void runDetectionTick();
      }, 850);
      void runDetectionTick();
    } catch (err) {
      cameraOnRef.current = false;
      personTracksRef.current.clear();
      trackSeqRef.current = 0;
      setCoverage({ tracked: 0, matched: 0, scanning: 0 });
      setCameraError(err instanceof Error ? err.message : "Failed to start camera");
    }
  }

  async function handleEnrollIdentity() {
    const name = identityName.trim();
    if (!name) {
      setIdentityStatus("identity: enter a name first");
      return;
    }
    const video = videoRef.current;
    if (!video || !cameraOnRef.current) {
      setIdentityStatus("identity: start camera first");
      return;
    }
    const person = detections
      .filter((d) => d.label === "person")
      .sort((a, b) => b.bbox[2] * b.bbox[3] - a.bbox[2] * a.bbox[3])[0];
    if (!person) {
      setIdentityStatus("identity: no person box to enroll");
      return;
    }
    setIdentityBusy(true);
    setIdentityStatus("identity: capturing samples...");
    try {
      const samples: string[] = [];
      for (let i = 0; i < 8; i += 1) {
        const sample = cropDetectionAsJpeg(video, person.bbox);
        if (sample) samples.push(sample);
        await new Promise((resolve) => window.setTimeout(resolve, 150));
      }
      if (!samples.length) throw new Error("Failed to capture samples");
      const enrolled = await enrollVisionIdentity(name, samples);
      setIdentityName("");
      setIdentityStatus(`identity: enrolled ${enrolled.display_name}`);
      await refreshIdentities();
    } catch (err) {
      setIdentityStatus(err instanceof Error ? "identity: " + err.message : "identity: enrollment failed");
    } finally {
      setIdentityBusy(false);
    }
  }

  async function handleDeleteIdentity(personId: string) {
    const pid = String(personId || "").trim();
    if (!pid) {
      setIdentityStatus("identity: invalid person id");
      return;
    }
    setIdentityBusy(true);
    setIdentityStatus("identity: deleting...");
    try {
      await deleteVisionIdentity(pid);
      setIdentities((rows) => rows.filter((r) => r.person_id !== pid));
      await refreshIdentities();
      setIdentityStatus("identity: deleted");
    } catch (err) {
      setIdentityStatus(err instanceof Error ? "identity: " + err.message : "identity: delete failed");
    } finally {
      setIdentityBusy(false);
    }
  }

  async function handleInterrupt() {
    setBusy(true);
    try {
      await onInterrupt();
    } finally {
      setBusy(false);
    }
  }

  async function handleStartUrl() {
    if (!sourceUrl.trim()) return;
    setBusy(true);
    try {
      await onStartUrlStream(sourceUrl.trim(), sourceType);
      setSourceUrl("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Motion.section
      className="controls glass"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
    >
      <div className="controlsHeader">
        <h3>Realtime Inputs</h3>
      </div>
      <div className="controlsGrid">
        <button className="btn warn" disabled={!connected || busy} onClick={handleInterrupt}>
          <ShieldAlert size={16} />
          Interrupt
        </button>
        <button className="btn" disabled={!connected} onClick={toggleCamera}>
          {cameraOn ? <CameraOff size={16} /> : <Camera size={16} />}
          {cameraOn ? "Stop Camera Stream" : "Start Camera Stream"}
        </button>
      </div>
      {cameraError ? <p className="hint error">{cameraError}</p> : null}
      <div className="cameraPanel">
        <video ref={videoRef} className="cameraPreview" playsInline muted />
        <canvas ref={overlayRef} className="cameraOverlay" />
      </div>
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <div className="controlsGrid">
        <button
          className="btn"
          disabled={!cameraOn}
          onClick={() =>
            setDetectionEnabled((v) => {
              const next = !v;
              detectionEnabledRef.current = next;
              return next;
            })
          }
        >
          {detectionEnabled ? "Disable Detection" : "Enable Detection"}
        </button>
        <button className="btn" disabled={!cameraOn} onClick={() => setRecognitionEnabled((v) => !v)}>
          {recognitionEnabled ? "Disable Identity" : "Enable Identity"}
        </button>
        <button className="btn" onClick={() => void refreshIdentities()}>
          Refresh Identities
        </button>
      </div>
      <p className="hint">{detectorStatus}</p>
      <p className="hint">{identityStatus}</p>
      <div className="detectionPills">
        <span className="detectionPill">Tracked: {coverage.tracked}</span>
        <span className="detectionPill">Matched: {coverage.matched}</span>
        <span className="detectionPill">Scanning: {coverage.scanning}</span>
      </div>
      <div className="streamRow">
        <input
          value={identityName}
          onChange={(e) => setIdentityName(e.target.value)}
          placeholder="Identity name (e.g. Jiten)"
        />
        <button className="btn" disabled={!cameraOn || identityBusy} onClick={handleEnrollIdentity}>
          Enroll Visible Person
        </button>
      </div>
      {identities.length ? (
        <div className="detectionPills">
          {identities.map((p) => (
            <button
              key={p.person_id}
              className="detectionPill"
              disabled={identityBusy}
              onClick={() => void handleDeleteIdentity(p.person_id)}
              title="Click to delete identity"
            >
              {p.display_name} ({p.sample_count})
            </button>
          ))}
        </div>
      ) : null}
      {detections.length ? (
        <div className="detectionPills">
          {detections.slice(0, 8).map((d, idx) => (
            <span key={`${d.label}-${idx}`} className="detectionPill">
              {d.identity ? `${d.identity} (${d.label})` : d.label} {Math.round(d.score * 100)}%
            </span>
          ))}
        </div>
      ) : (
        <p className="hint">No objects detected yet.</p>
      )}

      <div className="streamRow">
        <select value={sourceType} onChange={(e) => setSourceType(e.target.value as "rtsp" | "webrtc" | "http")}>
          <option value="http">HTTP snapshot URL</option>
          <option value="rtsp">RTSP URL</option>
          <option value="webrtc">WebRTC URL</option>
        </select>
        <input
          value={sourceUrl}
          onChange={(e) => setSourceUrl(e.target.value)}
          placeholder="rtsp://... or https://.../frame.jpg"
        />
        <button className="btn" disabled={!connected || busy} onClick={handleStartUrl}>
          <Radio size={16} />
          Start URL Stream
        </button>
      </div>
      <p className="hint">
        <Link2 size={12} />
        On iPhone Safari, open this page over HTTPS and tap Start Camera Stream.
      </p>
    </Motion.section>
  );
}
