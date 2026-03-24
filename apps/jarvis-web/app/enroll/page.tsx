"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { Building2, Camera, CameraOff, Globe2, RefreshCw, UserPlus, Users } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  deleteVisionIdentitySample,
  deleteVisionIdentity,
  enrollVisionIdentity,
  listVisionIdentitySamples,
  listVisionIdentities,
  type VisionIdentitySample,
  recognizeVisionIdentities,
  teachWorldConcept,
  type VisionIdentity
} from "../../lib/api";

type DetectionItem = {
  label: string;
  score: number;
  bbox: [number, number, number, number];
  identity?: string;
  identityScore?: number;
  personId?: string;
};

type IdentityLock = {
  bbox: [number, number, number, number];
  identity: string;
  identityScore: number;
  personId: string;
  lastSeenAt: number;
};

const COLORS = ["#22c55e", "#ef4444", "#f59e0b", "#3b82f6", "#06b6d4", "#a855f7"];

function colorForLabel(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) hash = (hash * 33 + label.charCodeAt(i)) >>> 0;
  return COLORS[hash % COLORS.length] || COLORS[0];
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

type CropRect = { x: number; y: number; w: number; h: number };

function clamp(val: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, val));
}

function clampRect(rect: CropRect, maxW: number, maxH: number): CropRect {
  const x = clamp(Math.floor(rect.x), 0, Math.max(0, maxW - 1));
  const y = clamp(Math.floor(rect.y), 0, Math.max(0, maxH - 1));
  const w = clamp(Math.floor(rect.w), 24, Math.max(24, maxW - x));
  const h = clamp(Math.floor(rect.h), 24, Math.max(24, maxH - y));
  return { x, y, w, h };
}

function padRect(rect: CropRect, px: number, py: number): CropRect {
  return {
    x: rect.x - rect.w * px,
    y: rect.y - rect.h * py,
    w: rect.w * (1 + px * 2),
    h: rect.h * (1 + py * 2)
  };
}

function cropRectToDataUrl(video: HTMLVideoElement, rect: CropRect, size = 192, quality = 0.84): string {
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;
  if (!vw || !vh) return "";
  const safe = clampRect(rect, vw, vh);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(video, safe.x, safe.y, safe.w, safe.h, 0, 0, size, size);
  return canvas.toDataURL("image/jpeg", quality);
}

function buildEnrollmentRects(bbox: [number, number, number, number]): { context: CropRect[]; detail: CropRect[] } {
  const base: CropRect = { x: bbox[0], y: bbox[1], w: bbox[2], h: bbox[3] };
  const context = [padRect(base, 0.22, 0.18), padRect(base, 0.12, 0.1)];
  const detailCore: CropRect[] = [
    padRect(base, 0.03, 0.03),
    { x: base.x + base.w * 0.1, y: base.y + base.h * 0.14, w: base.w * 0.8, h: base.h * 0.78 },
    { x: base.x + base.w * 0.12, y: base.y + base.h * 0.05, w: base.w * 0.76, h: base.h * 0.58 },
    { x: base.x + base.w * 0.28, y: base.y + base.h * 0.04, w: base.w * 0.44, h: base.h * 0.4 }
  ];
  const jitter = [0.0, 0.04, -0.04, 0.08, -0.08, 0.12];
  const detail = [
    ...detailCore,
    ...jitter.map((j) => ({
      x: base.x + base.w * (0.08 + j * 0.5),
      y: base.y + base.h * (0.1 + Math.abs(j) * 0.25),
      w: base.w * (0.82 - Math.abs(j) * 0.2),
      h: base.h * (0.8 - Math.abs(j) * 0.15)
    }))
  ];
  return { context, detail };
}

export default function EnrollPage() {
  const Motion = motion as any;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const detectTimerRef = useRef<number | null>(null);
  const modelRef = useRef<any>(null);
  const cameraOnRef = useRef(false);
  const detectBusyRef = useRef(false);
  const recognitionBusyRef = useRef(false);
  const recognitionAtRef = useRef(0);
  const identityLocksRef = useRef<Map<string, IdentityLock>>(new Map());

  const [cameraOn, setCameraOn] = useState(false);
  const [status, setStatus] = useState("detector: idle");
  const [error, setError] = useState("");
  const [detections, setDetections] = useState<DetectionItem[]>([]);
  const [selectedPersonIdx, setSelectedPersonIdx] = useState(0);
  const [identityName, setIdentityName] = useState("");
  const [roleContext, setRoleContext] = useState("");
  const [notes, setNotes] = useState("");
  const [tagsCsv, setTagsCsv] = useState("");
  const [saving, setSaving] = useState(false);
  const [recognitionEnabled, setRecognitionEnabled] = useState(true);
  const [identities, setIdentities] = useState<VisionIdentity[]>([]);
  const [activeIdentityId, setActiveIdentityId] = useState("");
  const [identitySamples, setIdentitySamples] = useState<VisionIdentitySample[]>([]);
  const [samplesBusy, setSamplesBusy] = useState(false);
  const [identityStatus, setIdentityStatus] = useState("identity: loading...");

  const personDetections = useMemo(() => detections.filter((d) => d.label === "person"), [detections]);
  const matchedNow = useMemo(() => {
    const seen = new Set<string>();
    const rows: Array<{ key: string; name: string; score: number }> = [];
    for (const d of personDetections) {
      const name = String(d.identity || "").trim();
      if (!name) continue;
      const personId = String(d.personId || name).trim();
      if (seen.has(personId)) continue;
      seen.add(personId);
      rows.push({ key: personId, name, score: Number(d.identityScore || 0) });
    }
    return rows;
  }, [personDetections]);
  const detectorState = status.replace(/^detector:\s*/i, "").trim() || "idle";

  useEffect(() => {
    if (!personDetections.length) {
      setSelectedPersonIdx(0);
      return;
    }
    if (selectedPersonIdx >= personDetections.length) {
      setSelectedPersonIdx(0);
    }
  }, [personDetections, selectedPersonIdx]);

  useEffect(() => {
    if (!recognitionEnabled) identityLocksRef.current.clear();
  }, [recognitionEnabled]);

  useEffect(() => {
    void refreshIdentities();
    return () => {
      if (detectTimerRef.current !== null) window.clearInterval(detectTimerRef.current);
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
    };
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

  async function openIdentitySamples(personId: string) {
    const pid = String(personId || "").trim();
    if (!pid) return;
    setActiveIdentityId(pid);
    setIdentitySamples([]);
    setSamplesBusy(true);
    setIdentityStatus("identity: loading samples...");
    try {
      const rows = await listVisionIdentitySamples(pid);
      setIdentitySamples(rows);
      setIdentityStatus(`identity: loaded ${rows.length} samples`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "samples unavailable";
      if (/404|not found/i.test(msg)) {
        setIdentityStatus("identity: samples API unavailable (restart backend to load latest routes)");
      } else {
        setIdentityStatus("identity: " + msg);
      }
    } finally {
      setSamplesBusy(false);
    }
  }

  async function ensureDetector() {
    if (modelRef.current) return modelRef.current;
    setStatus("detector: loading");
    const tf = await import("@tensorflow/tfjs");
    const cocoSsd = await import("@tensorflow-models/coco-ssd");
    await tf.ready();
    modelRef.current = await cocoSsd.load({ base: "lite_mobilenet_v2" });
    setStatus("detector: ready");
    return modelRef.current;
  }

  function draw(items: DetectionItem[]) {
    if (!videoRef.current || !overlayRef.current) return;
    const width = videoRef.current.videoWidth || 960;
    const height = videoRef.current.videoHeight || 540;
    const overlay = overlayRef.current;
    overlay.width = width;
    overlay.height = height;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    let personSeen = -1;
    items.forEach((item) => {
      const [x, y, w, h] = item.bbox;
      if (item.label === "person") personSeen += 1;
      const isSelected = item.label === "person" && personSeen === selectedPersonIdx;
      const color = isSelected ? "#facc15" : colorForLabel(item.identity || item.label);
      ctx.lineWidth = isSelected ? 4 : 3;
      ctx.strokeStyle = color;
      ctx.fillStyle = `${color}22`;
      ctx.strokeRect(x, y, w, h);
      ctx.fillRect(x, y, w, h);
      const idPct = typeof item.identityScore === "number" ? Math.round(item.identityScore * 100) : 0;
      const title = item.identity
        ? `${item.identity} (${item.label}) ${Math.round(item.score * 100)}% id:${idPct}%`
        : `${item.label} ${Math.round(item.score * 100)}%`;
      const textW = ctx.measureText(title).width + 10;
      const textY = Math.max(16, y - 4);
      ctx.fillStyle = "rgba(4, 15, 20, 0.86)";
      ctx.fillRect(x, textY - 14, textW, 16);
      ctx.fillStyle = color;
      ctx.fillText(title, x + 5, textY - 2);
    });
  }

  async function detectTick() {
    if (!cameraOnRef.current || detectBusyRef.current || !videoRef.current) return;
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
        .slice(0, 12);
      let resolved = mapped;
      const now = Date.now();
      const locks = identityLocksRef.current;
      for (const [key, lock] of locks.entries()) {
        if (now - lock.lastSeenAt > 10000) locks.delete(key);
      }
      const personWithIdx = mapped.map((d, idx) => ({ d, idx })).filter((x) => x.d.label === "person");
      const lockedIdx = new Set<number>();
      if (recognitionEnabled) {
        for (const { d, idx } of personWithIdx) {
          let best: { key: string; lock: IdentityLock; iou: number } | null = null;
          for (const [key, lock] of locks.entries()) {
            const iou = bboxIou(d.bbox, lock.bbox);
            if (iou < 0.45) continue;
            if (!best || iou > best.iou) best = { key, lock, iou };
          }
          if (!best) continue;
          lockedIdx.add(idx);
          const keep = best.lock;
          keep.bbox = d.bbox;
          keep.lastSeenAt = now;
          locks.set(best.key, keep);
          resolved[idx] = {
            ...resolved[idx],
            identity: keep.identity,
            identityScore: keep.identityScore,
            personId: keep.personId
          };
        }
      }
      const toRecognize = personWithIdx.filter(({ idx }) => !lockedIdx.has(idx));
      if (recognitionEnabled && toRecognize.length > 0 && !recognitionBusyRef.current) {
        if (now - recognitionAtRef.current > 900) {
          recognitionBusyRef.current = true;
          recognitionAtRef.current = now;
          try {
            const samples = toRecognize
              .map(({ d, idx }) => ({
                sample_id: "enroll-det-" + idx,
                detection_index: idx,
                image_b64: cropRectToDataUrl(video, { x: d.bbox[0], y: d.bbox[1], w: d.bbox[2], h: d.bbox[3] }, 176, 0.8)
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
                const lockKey = `${found.personId}:${idx}`;
                locks.set(lockKey, {
                  bbox: item.bbox,
                  identity: found.name,
                  identityScore: found.score,
                  personId: found.personId,
                  lastSeenAt: now
                });
                return {
                  ...item,
                  identity: found.name,
                  identityScore: found.score,
                  personId: found.personId
                };
              });
            }
          } catch {
            // keep detection resilient even when identity service is unavailable
          } finally {
            recognitionBusyRef.current = false;
          }
        }
      }
      setDetections(resolved);
      draw(resolved);
    } catch (err) {
      setStatus(err instanceof Error ? "detector: " + err.message : "detector: unavailable");
    } finally {
      detectBusyRef.current = false;
    }
  }

  async function toggleCamera() {
    if (cameraOnRef.current) {
      if (detectTimerRef.current !== null) {
        window.clearInterval(detectTimerRef.current);
        detectTimerRef.current = null;
      }
      const stream = videoRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      cameraOnRef.current = false;
      setCameraOn(false);
      setStatus("detector: idle");
      setDetections([]);
      identityLocksRef.current.clear();
      draw([]);
      return;
    }
    try {
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 1280, height: 720 },
        audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      cameraOnRef.current = true;
      setCameraOn(true);
      detectTimerRef.current = window.setInterval(() => void detectTick(), 500);
      void detectTick();
    } catch (err) {
      cameraOnRef.current = false;
      identityLocksRef.current.clear();
      setError(err instanceof Error ? err.message : "Failed to start camera");
    }
  }

  async function enrollSelected() {
    const name = identityName.trim();
    if (!name) {
      setIdentityStatus("identity: enter a name");
      return;
    }
    const selected = personDetections[selectedPersonIdx];
    if (!selected || !videoRef.current) {
      setIdentityStatus("identity: select a visible person first");
      return;
    }
    setSaving(true);
    setIdentityStatus("identity: capturing selected person (2 context + detailed samples)...");
    try {
      const samples: string[] = [];
      const plan = buildEnrollmentRects(selected.bbox);
      for (const rect of plan.context) {
        const img = cropRectToDataUrl(videoRef.current, rect, 200, 0.84);
        if (img) samples.push(img);
        await new Promise((resolve) => window.setTimeout(resolve, 90));
      }
      for (const rect of plan.detail) {
        const img = cropRectToDataUrl(videoRef.current, rect, 192, 0.86);
        if (img) samples.push(img);
        await new Promise((resolve) => window.setTimeout(resolve, 80));
      }
      const tags = tagsCsv
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      const metadata = {
        source: "enrollment_page",
        role_context: roleContext.trim(),
        notes: notes.trim(),
        tags,
        detected_labels: detections.slice(0, 8).map((d) => d.label),
        enrolled_at: new Date().toISOString()
      };
      const out = await enrollVisionIdentity(name, samples.slice(0, 12), metadata);
      setIdentityStatus(`identity: enrolled ${out.display_name}; syncing world knowledge...`);
      const worldTags = Array.from(
        new Set(
          [
            "person",
            "identity",
            "enrollment",
            ...tags,
            ...roleContext
              .split(",")
              .map((t) => t.trim().toLowerCase())
              .filter(Boolean)
          ].filter(Boolean)
        )
      );
      const worldNotes = [
        `Identity enrolled from live feed for ${out.display_name}.`,
        roleContext.trim() ? `Role/Context: ${roleContext.trim()}` : "",
        notes.trim() ? `Notes: ${notes.trim()}` : ""
      ]
        .filter(Boolean)
        .join(" ");
      try {
        await teachWorldConcept({
          topic: out.display_name,
          notes: worldNotes,
          tags: worldTags,
          detections: [
            {
              label: "person",
              confidence: Math.round((selected.score || 0) * 100) / 100,
              identity_name: out.display_name,
              person_id: out.person_id,
              source: "enrollment_studio"
            }
          ],
          metadata: {
            source: "enrollment_studio",
            linked_identity: true,
            person_id: out.person_id,
            sample_count: out.sample_count,
            role_context: roleContext.trim(),
            enrolled_at: new Date().toISOString()
          },
          enrich_web: false
        });
        setIdentityStatus(`identity: enrolled ${out.display_name} + world knowledge linked`);
      } catch (worldErr) {
        setIdentityStatus(
          worldErr instanceof Error
            ? `identity: enrolled ${out.display_name}; world sync failed (${worldErr.message})`
            : `identity: enrolled ${out.display_name}; world sync failed`
        );
      }
      setIdentityName("");
      await refreshIdentities();
    } catch (err) {
      setIdentityStatus(err instanceof Error ? "identity: " + err.message : "identity: enrollment failed");
    } finally {
      setSaving(false);
    }
  }

  async function deleteIdentity(personId: string) {
    const pid = String(personId || "").trim();
    if (!pid) {
      setIdentityStatus("identity: invalid person id");
      return;
    }
    setSaving(true);
    setIdentityStatus("identity: deleting...");
    try {
      await deleteVisionIdentity(pid);
      setIdentities((rows) => rows.filter((r) => r.person_id !== pid));
      if (activeIdentityId === pid) {
        setActiveIdentityId("");
        setIdentitySamples([]);
      }
      await refreshIdentities();
      setIdentityStatus("identity: deleted");
    } catch (err) {
      setIdentityStatus(err instanceof Error ? "identity: " + err.message : "identity: delete failed");
    } finally {
      setSaving(false);
    }
  }

  async function deleteSample(personId: string, sampleId: string) {
    setSamplesBusy(true);
    try {
      await deleteVisionIdentitySample(personId, sampleId);
      const rows = await listVisionIdentitySamples(personId);
      setIdentitySamples(rows);
      await refreshIdentities();
      setIdentityStatus(`identity: sample deleted (${rows.length} remaining)`);
    } catch (err) {
      setIdentityStatus(err instanceof Error ? "identity: " + err.message : "identity: sample delete failed");
    } finally {
      setSamplesBusy(false);
    }
  }

  return (
    <main className="enrollShell enrollBusiness">
      <Motion.section
        className="enrollMain enrollFrame"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: "easeOut" }}
      >
        <div className="enrollHeader enrollTopBar">
          <div className="enrollTitleBlock">
            <p className="enrollEyebrow">
              <Building2 size={14} />
              Identity Operations
            </p>
            <h1>Vision Enrollment Studio</h1>
            <p>Enroll people from live feed, attach business context, and keep registry clean.</p>
          </div>
          <div className="controlsGrid">
            <Link href="/" className="enrollBizBtn enrollBtnGhost">
              Live Console
            </Link>
            <Link href="/world-teaching" className="enrollBizBtn enrollBtnGhost">
              <Globe2 size={14} />
              World Teaching
            </Link>
          </div>
        </div>

        <div className="enrollStatsRow">
          <article className="enrollStatCard">
            <span className="enrollStatLabel">Registered Identities</span>
            <strong className="enrollStatValue">{identities.length}</strong>
          </article>
          <article className="enrollStatCard">
            <span className="enrollStatLabel">Visible Persons</span>
            <strong className="enrollStatValue">{personDetections.length}</strong>
          </article>
          <article className="enrollStatCard">
            <span className="enrollStatLabel">Detector State</span>
            <strong className="enrollStatValue">{detectorState}</strong>
          </article>
        </div>

        <div className="controlsGrid enrollToolbar">
          <button className="enrollBizBtn" onClick={toggleCamera}>
            {cameraOn ? <CameraOff size={16} /> : <Camera size={16} />}
            {cameraOn ? "Stop Camera" : "Start Camera"}
          </button>
          <button className="enrollBizBtn enrollBtnGhost" onClick={() => void refreshIdentities()}>
            <RefreshCw size={16} />
            Refresh Identities
          </button>
          <button className="enrollBizBtn enrollBtnGhost" disabled={!cameraOn} onClick={() => setRecognitionEnabled((v) => !v)}>
            {recognitionEnabled ? "Disable Match Labels" : "Enable Match Labels"}
          </button>
          <span className="enrollBadge">
            <Users size={12} />
            {identities.length} enrolled
          </span>
        </div>

        {error ? <p className="hint error">{error}</p> : <p className="hint">{status}</p>}
        <p className="hint">{identityStatus}</p>

        <div className="enrollGrid enrollWorkbench">
          <section className="enrollCard enrollFeedCard">
            <div className="enrollCardHeader">
              <h3>Live Feed</h3>
              <span className="enrollBadge">{cameraOn ? "camera live" : "camera off"}</span>
            </div>
            <div className="cameraPanel enrollCameraPanel">
              <video ref={videoRef} className="cameraPreview" playsInline muted />
              <canvas ref={overlayRef} className="cameraOverlay" />
            </div>
            <div className="enrollPickerRow">
              <label htmlFor="person-select">Enrollment Target</label>
              <select
                id="person-select"
                value={String(selectedPersonIdx)}
                onChange={(e) => setSelectedPersonIdx(Number(e.target.value || 0))}
                disabled={!personDetections.length}
              >
                {personDetections.length ? (
                  personDetections.map((d, idx) => (
                    <option key={`${idx}-${d.score}`} value={idx}>
                      {d.identity ? `${d.identity} • ` : `Person #${idx + 1} • `}
                      {Math.round(d.score * 100)}%
                    </option>
                  ))
                ) : (
                  <option value="0">No person detected</option>
                )}
              </select>
            </div>
            <p className="hint">Use front-facing posture and stable lighting for best enrollment quality.</p>
            <p className="hint">Sample strategy: 2 context frames + detailed crops for robust identity under style changes.</p>
            {matchedNow.length ? (
              <div className="detectionPills">
                {matchedNow.map((m) => (
                  <span key={m.key} className="detectionPill">
                    Matched now: {m.name} ({Math.round(m.score * 100)}%)
                  </span>
                ))}
              </div>
            ) : (
              <p className="hint">Matched now: none</p>
            )}
          </section>

          <section className="enrollPanelStack">
            <article className="enrollCard">
              <h3>Identity Details</h3>
              <div className="enrollFields">
                <input
                  value={identityName}
                  onChange={(e) => setIdentityName(e.target.value)}
                  placeholder="Full name (e.g. Jiten Sabharwal)"
                />
                <input
                  value={roleContext}
                  onChange={(e) => setRoleContext(e.target.value)}
                  placeholder="Role / relation (e.g. Founder, Sales Lead)"
                />
                <input
                  value={tagsCsv}
                  onChange={(e) => setTagsCsv(e.target.value)}
                  placeholder="Tags (comma separated)"
                />
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Business context notes for Jarvis..."
                  rows={3}
                />
              </div>
              <button className="enrollBizBtn enrollBtnPrimary btnWide" disabled={saving || !cameraOn} onClick={enrollSelected}>
                <UserPlus size={16} />
                Enroll Selected Person
              </button>
            </article>

            <article className="enrollCard">
              <div className="enrollCardHeader">
                <h3>Identity Registry</h3>
                <span className="enrollBadge">{identities.length} total</span>
              </div>
              {identities.length ? (
                <div className="enrollIdentityList">
                  {identities.map((item) => (
                    <div key={item.person_id} className="enrollIdentityRow">
                      <div>
                        <p className="enrollIdentityName">{item.display_name}</p>
                        <p className="enrollIdentityMeta">{item.sample_count} samples</p>
                      </div>
                      <button
                        className="enrollBizBtn enrollBtnDanger"
                        disabled={saving}
                        onClick={() => void deleteIdentity(item.person_id)}
                        title="Delete identity"
                      >
                        Delete
                      </button>
                      <button
                        className="enrollBizBtn enrollBtnGhost"
                        disabled={samplesBusy}
                        onClick={() => void openIdentitySamples(item.person_id)}
                        title="View stored samples"
                      >
                        Samples
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="hint">No identities enrolled yet.</p>
              )}
              {activeIdentityId && samplesBusy ? (
                <p className="hint" style={{ marginTop: 8 }}>
                  Loading samples...
                </p>
              ) : null}
              {activeIdentityId && identitySamples.length ? (
                <div className="enrollIdentityList" style={{ marginTop: 10 }}>
                  {identitySamples.map((s) => (
                    <div key={s.sample_id} className="enrollIdentityRow">
                      <img
                        src={s.image_b64}
                        alt={"sample-" + s.sample_id}
                        style={{ width: 84, height: 84, borderRadius: 8, objectFit: "cover", border: "1px solid rgba(130,170,210,0.35)" }}
                      />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <p className="enrollIdentityMeta" style={{ margin: 0 }}>
                          {new Date(s.created_at * 1000).toLocaleString()}
                        </p>
                        <p className="enrollIdentityMeta" style={{ margin: "4px 0 0" }}>
                          {s.sample_id}
                        </p>
                      </div>
                      <button
                        className="enrollBizBtn enrollBtnDanger"
                        disabled={samplesBusy}
                        onClick={() => void deleteSample(activeIdentityId, s.sample_id)}
                      >
                        Delete Sample
                      </button>
                    </div>
                  ))}
                </div>
              ) : activeIdentityId ? (
                <p className="hint" style={{ marginTop: 8 }}>
                  No stored samples available for this identity.
                </p>
              ) : null}
            </article>
          </section>
        </div>

        {/*
          Keep the legacy chips for quick visual scan while preserving new business layout.
        */}
        {identities.length ? (
          <div className="detectionPills">
            {identities.map((item) => (
              <span key={item.person_id} className="detectionPill">
                {item.display_name} ({item.sample_count})
              </span>
            ))}
          </div>
        ) : null}
      </Motion.section>
    </main>
  );
}
