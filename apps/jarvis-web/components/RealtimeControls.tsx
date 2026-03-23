"use client";

import { motion } from "framer-motion";
import { Camera, CameraOff, Link2, Radio, ShieldAlert } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

export type DetectionItem = {
  label: string;
  score: number;
  bbox: [number, number, number, number];
};

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
  const modelRef = useRef<any>(null);
  const detectionPushAtRef = useRef(0);
  const [cameraOn, setCameraOn] = useState(false);
  const [busy, setBusy] = useState(false);
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceType, setSourceType] = useState<"rtsp" | "webrtc" | "http">("http");
  const [cameraError, setCameraError] = useState("");
  const [detectorStatus, setDetectorStatus] = useState("detector: idle");
  const [detections, setDetections] = useState<DetectionItem[]>([]);
  const [detectionEnabled, setDetectionEnabled] = useState(true);

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
    ctx.lineWidth = 2;
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    for (const item of items) {
      const [x, y, w, h] = item.bbox;
      const scorePct = Math.round(item.score * 100);
      const title = `${item.label} ${scorePct}%`;
      ctx.strokeStyle = "rgba(75, 248, 210, 0.95)";
      ctx.fillStyle = "rgba(75, 248, 210, 0.16)";
      ctx.strokeRect(x, y, w, h);
      ctx.fillRect(x, y, w, h);
      const textW = ctx.measureText(title).width + 10;
      const textY = Math.max(16, y - 4);
      ctx.fillStyle = "rgba(4, 15, 20, 0.88)";
      ctx.fillRect(x, textY - 14, textW, 16);
      ctx.fillStyle = "rgba(195, 255, 244, 1)";
      ctx.fillText(title, x + 5, textY - 2);
    }
  }

  async function runDetectionTick() {
    if (!detectionEnabled || detectBusyRef.current || !videoRef.current || !cameraOn) return;
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight) return;
    detectBusyRef.current = true;
    try {
      const model = await ensureDetector();
      const raw = await model.detect(video, 10, 0.45);
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
      setDetections(mapped);
      drawDetections(mapped);
      if (onDetections && Date.now() - detectionPushAtRef.current > 1200) {
        detectionPushAtRef.current = Date.now();
        await onDetections(mapped);
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
      drawDetections([]);
      setCameraOn(false);
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
      }, 1400);
      detectTimerRef.current = window.setInterval(() => {
        void runDetectionTick();
      }, 550);
      void runDetectionTick();
      setCameraOn(true);
    } catch (err) {
      setCameraError(err instanceof Error ? err.message : "Failed to start camera");
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
        <button className="btn" disabled={!cameraOn} onClick={() => setDetectionEnabled((v) => !v)}>
          {detectionEnabled ? "Disable Detection" : "Enable Detection"}
        </button>
      </div>
      <p className="hint">{detectorStatus}</p>
      {detections.length ? (
        <div className="detectionPills">
          {detections.slice(0, 8).map((d, idx) => (
            <span key={`${d.label}-${idx}`} className="detectionPill">
              {d.label} {Math.round(d.score * 100)}%
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
