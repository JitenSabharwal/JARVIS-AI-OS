"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { BookOpenCheck, Globe2, PencilLine, RefreshCw, Sparkles, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  addWorldConceptLink,
  deleteWorldConcept,
  deleteWorldConceptLink,
  enrichWorldConcept,
  getWorldConcept,
  listWorldConcepts,
  logWorldConceptLinkInteraction,
  runWorldConceptLinkBrowserUse,
  teachWorldConcept,
  updateWorldConcept,
  type WorldConcept
} from "../../lib/api";

function asCsv(items: string[]): string {
  return (Array.isArray(items) ? items : []).join(", ");
}

function formatTs(ts: number): string {
  if (!Number.isFinite(ts)) return "-";
  return new Date(ts * 1000).toLocaleString();
}

function humanAction(action: string): string {
  return String(action || "")
    .split("_")
    .filter(Boolean)
    .join(" ");
}

function pct(part: number, total: number): number {
  if (!Number.isFinite(part) || !Number.isFinite(total) || total <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

export default function WorldTeachingPage() {
  const Motion = motion as any;
  const [topic, setTopic] = useState("");
  const [notes, setNotes] = useState("");
  const [tagsCsv, setTagsCsv] = useState("");
  const [labelsCsv, setLabelsCsv] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("world: loading...");
  const [activeAction, setActiveAction] = useState("");
  const [activeConceptId, setActiveConceptId] = useState("");
  const [activeLinkId, setActiveLinkId] = useState("");
  const [concepts, setConcepts] = useState<WorldConcept[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [selected, setSelected] = useState<WorldConcept | null>(null);

  const [editTopic, setEditTopic] = useState("");
  const [editTagsCsv, setEditTagsCsv] = useState("");
  const [editNote, setEditNote] = useState("");

  const [linkUrl, setLinkUrl] = useState("");
  const [linkTitle, setLinkTitle] = useState("");
  const [linkNotes, setLinkNotes] = useState("");
  const [linkTagsCsv, setLinkTagsCsv] = useState("");
  const [interactionSummary, setInteractionSummary] = useState("");
  const [interactionFacts, setInteractionFacts] = useState("");
  const [interactionPattern, setInteractionPattern] = useState("");
  const [interactionOutcome, setInteractionOutcome] = useState("useful");

  const totals = useMemo(
    () => ({
      concepts: concepts.length,
      webFacts: concepts.reduce((n, c) => n + (c.web_facts_count || 0), 0),
      links: concepts.reduce((n, c) => n + (c.reference_links_count || 0), 0)
    }),
    [concepts]
  );

  const progress = useMemo(() => {
    if (!selected) {
      return {
        score: 0,
        strength: 0,
        profile: 0,
        evidence: 0,
        validation: 0
      };
    }
    const profile = Math.round(
      Math.min(1, (selected.notes_count || 0) / 4) * 60 + Math.min(1, (selected.detections_count || 0) / 8) * 40
    );
    const evidence = Math.round(
      Math.min(1, (selected.reference_links_count || 0) / 5) * 55 + Math.min(1, (selected.web_facts_count || 0) / 12) * 45
    );
    const usefulRuns = (selected.learning_runs || []).filter((r) => r.outcome === "useful").length;
    const runCount = (selected.learning_runs || []).length;
    const validation = Math.round(
      Math.min(1, (selected.interaction_logs_count || 0) / 8) * 50 +
        Math.min(1, runCount / 8) * 30 +
        pct(usefulRuns, Math.max(1, runCount)) * 0.2
    );
    const strength = Math.round(profile * 0.35 + evidence * 0.4 + validation * 0.25);
    const score = Math.round(profile * 0.4 + evidence * 0.4 + validation * 0.2);
    return { score, strength, profile, evidence, validation };
  }, [selected]);

  useEffect(() => {
    void refreshConcepts();
  }, []);

  function startAction(action: string, conceptId = "", linkId = "") {
    setBusy(true);
    setActiveAction(action);
    setActiveConceptId(conceptId);
    setActiveLinkId(linkId);
  }

  function endAction() {
    setBusy(false);
    setActiveAction("");
    setActiveConceptId("");
    setActiveLinkId("");
  }

  async function refreshConcepts(selectConceptId?: string) {
    try {
      const rows = await listWorldConcepts(100);
      setConcepts(rows);
      setStatus(`world: ${rows.length} concepts`);
      const nextId = selectConceptId || selectedId;
      if (!nextId) {
        if (rows.length > 0) {
          await loadConcept(rows[0].concept_id);
        }
        return;
      }
      const found = rows.find((r) => r.concept_id === nextId);
      if (!found) {
        setSelectedId("");
        setSelected(null);
        if (rows.length > 0) {
          await loadConcept(rows[0].concept_id);
        }
        return;
      }
      await loadConcept(nextId);
    } catch (err) {
      setStatus(err instanceof Error ? "world: " + err.message : "world: unavailable");
    }
  }

  async function loadConcept(conceptId: string) {
    setSelectedId(conceptId);
    const local = concepts.find((c) => c.concept_id === conceptId) || null;
    if (local) {
      setSelected(local);
      setEditTopic(local.topic || "");
      setEditTagsCsv(asCsv(local.tags || []));
    }
    try {
      const row = await getWorldConcept(conceptId);
      setSelected(row);
      setEditTopic(row.topic || "");
      setEditTagsCsv(asCsv(row.tags || []));
    } catch (err) {
      if (!local) {
        setStatus(err instanceof Error ? "world: " + err.message : "world: failed to load concept");
      } else {
        setStatus("world: opened local concept view (detail sync unavailable)");
      }
    }
  }

  async function teach() {
    const cleanTopic = topic.trim();
    if (!cleanTopic) {
      setStatus("world: topic is required");
      return;
    }
    startAction("teach_new_concept");
    setStatus("world: teaching and enriching...");
    try {
      const tags = tagsCsv
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      const detections = labelsCsv
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean)
        .map((label) => ({ label }));
      const learned = await teachWorldConcept({
        topic: cleanTopic,
        notes: notes.trim(),
        tags,
        detections,
        enrich_web: true,
        max_items: 6,
        metadata: { source: "world_teaching_page" }
      });
      setTopic("");
      setNotes("");
      setTagsCsv("");
      setLabelsCsv("");
      setStatus(`world: taught ${learned.topic}`);
      await refreshConcepts(learned.concept_id);
    } catch (err) {
      setStatus(err instanceof Error ? "world: " + err.message : "world: teach failed");
    } finally {
      endAction();
    }
  }

  async function enrich(conceptId: string, conceptTopic: string) {
    startAction("enrich_concept", conceptId);
    setStatus(`world: enriching ${conceptTopic}...`);
    try {
      await enrichWorldConcept(conceptId, 6);
      await refreshConcepts(conceptId);
      setStatus(`world: ${conceptTopic} enriched`);
    } catch (err) {
      setStatus(err instanceof Error ? "world: " + err.message : "world: enrich failed");
    } finally {
      endAction();
    }
  }

  async function saveConceptPatch() {
    if (!selectedId) return;
    startAction("save_concept_patch", selectedId);
    try {
      const tags = editTagsCsv
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      await updateWorldConcept(selectedId, {
        topic: editTopic.trim(),
        notes: editNote.trim() || undefined,
        tags
      });
      setEditNote("");
      await refreshConcepts(selectedId);
      setStatus("world: concept updated");
    } catch (err) {
      setStatus(err instanceof Error ? "world: " + err.message : "world: update failed");
    } finally {
      endAction();
    }
  }

  async function removeConcept() {
    if (!selectedId) return;
    startAction("delete_concept", selectedId);
    try {
      await deleteWorldConcept(selectedId);
      setSelectedId("");
      setSelected(null);
      await refreshConcepts();
      setStatus("world: concept deleted");
    } catch (err) {
      setStatus(err instanceof Error ? "world: " + err.message : "world: delete failed");
    } finally {
      endAction();
    }
  }

  async function addLink() {
    if (!selectedId) return;
    const url = linkUrl.trim();
    if (!url) {
      setStatus("world: link URL is required");
      return;
    }
    startAction("add_link", selectedId);
    try {
      const tags = linkTagsCsv
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      const facts = interactionFacts
        .split("\n")
        .map((t) => t.trim())
        .filter(Boolean);
      await addWorldConceptLink(selectedId, {
        url,
        title: linkTitle.trim(),
        notes: linkNotes.trim(),
        source_type: "browser_use_seed",
        tags,
        interaction: interactionSummary.trim()
          ? {
              summary: interactionSummary.trim(),
              extracted_facts: facts,
              pattern_hint: interactionPattern.trim(),
              outcome: interactionOutcome
            }
          : undefined
      });
      setLinkUrl("");
      setLinkTitle("");
      setLinkNotes("");
      setLinkTagsCsv("");
      setInteractionSummary("");
      setInteractionFacts("");
      setInteractionPattern("");
      setInteractionOutcome("useful");
      await refreshConcepts(selectedId);
      setStatus("world: link added");
    } catch (err) {
      setStatus(err instanceof Error ? "world: " + err.message : "world: add link failed");
    } finally {
      endAction();
    }
  }

  async function removeLink(linkId: string) {
    if (!selectedId) return;
    startAction("delete_link", selectedId, linkId);
    try {
      await deleteWorldConceptLink(selectedId, linkId);
      await refreshConcepts(selectedId);
      setStatus("world: link deleted");
    } catch (err) {
      setStatus(err instanceof Error ? "world: " + err.message : "world: link delete failed");
    } finally {
      endAction();
    }
  }

  async function autoLearnLink(linkId: string) {
    if (!selectedId) return;
    startAction("auto_learn_link", selectedId, linkId);
    try {
      await runWorldConceptLinkBrowserUse(selectedId, linkId, { max_items: 6, run_adapters: true });
      await refreshConcepts(selectedId);
      setStatus("world: auto-learn completed");
    } catch (err) {
      setStatus(err instanceof Error ? "world: " + err.message : "world: auto-learn failed");
    } finally {
      endAction();
    }
  }

  async function logInteraction(linkId: string) {
    if (!selectedId || !interactionSummary.trim()) {
      setStatus("world: write interaction summary first");
      return;
    }
    startAction("log_interaction", selectedId, linkId);
    try {
      await logWorldConceptLinkInteraction(selectedId, linkId, {
        summary: interactionSummary.trim(),
        extracted_facts: interactionFacts,
        pattern_hint: interactionPattern.trim(),
        outcome: interactionOutcome
      });
      setInteractionSummary("");
      setInteractionFacts("");
      setInteractionPattern("");
      await refreshConcepts(selectedId);
      setStatus("world: interaction logged");
    } catch (err) {
      setStatus(err instanceof Error ? "world: " + err.message : "world: interaction log failed");
    } finally {
      endAction();
    }
  }

  return (
    <main className="enrollShell enrollBusiness worldTeachShell">
      <Motion.section
        className="enrollMain enrollFrame"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: "easeOut" }}
      >
        <div className="enrollHeader enrollTopBar">
          <div className="enrollTitleBlock">
            <p className="enrollEyebrow">
              <Globe2 size={14} />
              Knowledge Operations
            </p>
            <h1>World Teaching Studio</h1>
            <p>Progressive workflow to teach, evidence, and validate real-world concepts.</p>
          </div>
          <div className="controlsGrid">
            <Link href="/" className="enrollBizBtn enrollBtnGhost">
              Live Console
            </Link>
            <Link href="/enroll" className="enrollBizBtn enrollBtnGhost">
              Enrollment Studio
            </Link>
          </div>
        </div>

        <div className="enrollStatsRow">
          <article className="enrollStatCard">
            <span className="enrollStatLabel">Concepts</span>
            <strong className="enrollStatValue">{totals.concepts}</strong>
          </article>
          <article className="enrollStatCard">
            <span className="enrollStatLabel">Web Facts</span>
            <strong className="enrollStatValue">{totals.webFacts}</strong>
          </article>
          <article className="enrollStatCard">
            <span className="enrollStatLabel">Reference Links</span>
            <strong className="enrollStatValue">{totals.links}</strong>
          </article>
        </div>

        <section className="enrollCard worldTeachCard">
          <div className="enrollCardHeader">
            <h3>Teach New Concept</h3>
            <button className="enrollBizBtn enrollBtnGhost" disabled={busy} onClick={() => void refreshConcepts()}>
              <RefreshCw size={14} />
              Refresh
            </button>
          </div>
          <div className="enrollFields">
            <input value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="Topic (e.g. climate risk)" />
            <input value={tagsCsv} onChange={(e) => setTagsCsv(e.target.value)} placeholder="Tags (comma separated)" />
            <input
              value={labelsCsv}
              onChange={(e) => setLabelsCsv(e.target.value)}
              placeholder="Related detections (e.g. car,person,traffic light)"
            />
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Key notes, definitions, common misconceptions, business relevance..."
              rows={3}
            />
          </div>
          <div className="controlsGrid">
            <button className="enrollBizBtn enrollBtnPrimary" disabled={busy} onClick={() => void teach()}>
              <BookOpenCheck size={16} />
              {activeAction === "teach_new_concept" ? "Teaching..." : "Teach + Web Enrich"}
            </button>
            <span className="enrollBadge">
              <Sparkles size={12} />
              progressive disclosure workflow
            </span>
          </div>
          <p className="hint">{status}</p>
          {busy ? (
            <p className="hint worldBusyPill">
              Working on {activeConceptId ? `concept ${activeConceptId}` : "knowledge library"}
              {activeLinkId ? ` • link ${activeLinkId}` : ""}
            </p>
          ) : null}
        </section>

        <section className="enrollCard worldTeachCard">
          <div className="enrollCardHeader">
            <h3>Knowledge Library</h3>
            <span className="enrollBadge">Select a concept to open Progressive Workflow below</span>
          </div>
          {concepts.length ? (
            <div className="worldConceptGrid">
              {concepts.map((c) => (
                <article
                  key={c.concept_id}
                  className={"liveWorldConceptCard" + (selectedId === c.concept_id ? " worldConceptCardActive" : "")}
                >
                  <div className="liveWorldConceptTop">
                    <strong>{c.topic}</strong>
                    <div className="controlsGrid">
                      <button className="enrollBizBtn enrollBtnGhost" disabled={busy} onClick={() => void loadConcept(c.concept_id)}>
                        <PencilLine size={14} />
                        {selectedId === c.concept_id ? "Managing" : "Manage"}
                      </button>
                      <button className="enrollBizBtn enrollBtnGhost" disabled={busy} onClick={() => void enrich(c.concept_id, c.topic)}>
                        {activeAction === "enrich_concept" && activeConceptId === c.concept_id ? "Enriching..." : "Enrich"}
                      </button>
                    </div>
                  </div>
                  <p>{c.latest_note || "No note yet."}</p>
                  <span className="liveWorldMeta">
                    {c.web_facts_count} web facts • {c.reference_links_count || 0} links • {c.detections_count} detections
                  </span>
                  {busy && activeConceptId === c.concept_id ? (
                    <span className="enrollBadge worldBusyPill">Active: {humanAction(activeAction)}</span>
                  ) : null}
                </article>
              ))}
            </div>
          ) : (
            <p className="hint">No concepts yet.</p>
          )}
        </section>

        {selected ? (
          <section className="enrollCard worldTeachCard">
            <div className="enrollCardHeader">
              <h3>Manage Concept: {selected.topic}</h3>
              <button className="enrollBizBtn enrollBtnDanger" disabled={busy} onClick={() => void removeConcept()}>
                <Trash2 size={14} />
                Delete Concept
              </button>
            </div>

            <div className="worldProgressCard">
              <div className="worldProgressHead">
                <strong>Completion: {progress.score}%</strong>
                <span className="enrollBadge">Knowledge strength: {progress.strength}%</span>
              </div>
              <div className="worldProgressTrack">
                <span className="worldProgressFill" style={{ width: `${progress.score}%` }} />
              </div>
              <div className="worldProgressSplit">
                <span>Profile {progress.profile}%</span>
                <span>Evidence {progress.evidence}%</span>
                <span>Validation {progress.validation}%</span>
              </div>
            </div>

            <details className="worldStep" open>
              <summary>Step 1: Build core profile ({progress.profile}%)</summary>
              <p className="hint">Define topic, add notes, and keep context tags clean.</p>
              <div className="enrollFields">
                <input value={editTopic} onChange={(e) => setEditTopic(e.target.value)} placeholder="Concept topic" />
                <input value={editTagsCsv} onChange={(e) => setEditTagsCsv(e.target.value)} placeholder="Tags (comma separated)" />
                <textarea
                  value={editNote}
                  onChange={(e) => setEditNote(e.target.value)}
                  rows={2}
                  placeholder="Add a new note while updating"
                />
              </div>
              <div className="controlsGrid">
                <button className="enrollBizBtn enrollBtnPrimary" disabled={busy} onClick={() => void saveConceptPatch()}>
                  {activeAction === "save_concept_patch" && activeConceptId === selectedId ? "Saving..." : "Save Changes"}
                </button>
                <span className="enrollBadge">{selected.notes_count} notes • {selected.detections_count} detections</span>
              </div>
            </details>

            <details className="worldStep" open={progress.profile >= 30}>
              <summary>Step 2: Add source evidence ({progress.evidence}%)</summary>
              <p className="hint">Attach trusted links and run auto-learn for evidence-backed enrichment.</p>
              <div className="enrollFields">
                <input value={linkUrl} onChange={(e) => setLinkUrl(e.target.value)} placeholder="https://..." />
                <input value={linkTitle} onChange={(e) => setLinkTitle(e.target.value)} placeholder="Link title" />
                <input value={linkTagsCsv} onChange={(e) => setLinkTagsCsv(e.target.value)} placeholder="Link tags (comma separated)" />
                <textarea
                  value={linkNotes}
                  onChange={(e) => setLinkNotes(e.target.value)}
                  rows={2}
                  placeholder="Why this link helps this concept"
                />
              </div>
              <div className="controlsGrid">
                <button className="enrollBizBtn enrollBtnPrimary" disabled={busy} onClick={() => void addLink()}>
                  Add Web Link Memory
                </button>
              </div>

              {selected.reference_links?.length ? (
                <div className="worldFactList">
                  {selected.reference_links.map((l) => (
                    <div key={l.link_id} className="worldLinkRow">
                      <a href={l.url} target="_blank" rel="noreferrer">
                        {l.title || l.url}
                      </a>
                      <div className="controlsGrid" style={{ marginTop: 6 }}>
                        <button className="enrollBizBtn enrollBtnPrimary" disabled={busy} onClick={() => void autoLearnLink(l.link_id)}>
                          {activeAction === "auto_learn_link" && activeLinkId === l.link_id ? "Auto Learning..." : "Auto Learn"}
                        </button>
                        <button className="enrollBizBtn enrollBtnGhost" disabled={busy} onClick={() => void removeLink(l.link_id)}>
                          {activeAction === "delete_link" && activeLinkId === l.link_id ? "Deleting..." : "Delete Link"}
                        </button>
                      </div>
                      {busy && activeLinkId === l.link_id ? (
                        <p className="hint worldBusyPill">This link is currently being processed.</p>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="hint">No reference links yet.</p>
              )}
            </details>

            <details className="worldStep" open={progress.evidence >= 30}>
              <summary>Step 3: Validate and strengthen ({progress.validation}%)</summary>
              <p className="hint">Log what worked in live research so future retrieval gets stronger.</p>
              <div className="enrollFields">
                <textarea
                  value={interactionSummary}
                  onChange={(e) => setInteractionSummary(e.target.value)}
                  rows={2}
                  placeholder="Summary from live browser-use interaction"
                />
                <textarea
                  value={interactionFacts}
                  onChange={(e) => setInteractionFacts(e.target.value)}
                  rows={2}
                  placeholder="Extracted facts (one per line)"
                />
                <input
                  value={interactionPattern}
                  onChange={(e) => setInteractionPattern(e.target.value)}
                  placeholder="Pattern hint for similar future searches"
                />
                <select value={interactionOutcome} onChange={(e) => setInteractionOutcome(e.target.value)}>
                  <option value="useful">useful</option>
                  <option value="partial">partial</option>
                  <option value="failed">failed</option>
                </select>
              </div>

              {selected.reference_links?.length ? (
                <div className="worldFactList">
                  {selected.reference_links.map((l) => (
                    <div key={`${l.link_id}-log`} className="worldLinkRow">
                      <a href={l.url} target="_blank" rel="noreferrer">
                        {l.title || l.url}
                      </a>
                      <div className="controlsGrid" style={{ marginTop: 6 }}>
                        <button className="enrollBizBtn enrollBtnGhost" disabled={busy} onClick={() => void logInteraction(l.link_id)}>
                          {activeAction === "log_interaction" && activeLinkId === l.link_id ? "Logging..." : "Log Interaction"}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}

              {selected.interaction_logs?.length ? (
                <div className="worldFactList">
                  {selected.interaction_logs.slice(0, 8).map((log) => (
                    <div key={log.log_id}>
                      <strong>{log.outcome || "logged"}</strong>
                      <p className="hint">{log.summary}</p>
                    </div>
                  ))}
                </div>
              ) : null}
            </details>

            {selected.learning_runs?.length ? (
              <div className="worldFactList">
                <strong>Learning Run History</strong>
                <div style={{ overflowX: "auto" }}>
                  <table className="worldRunsTable">
                    <thead>
                      <tr>
                        <th align="left">Time</th>
                        <th align="left">Link</th>
                        <th align="left">Query</th>
                        <th align="left">Facts</th>
                        <th align="left">Outcome</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.learning_runs.slice(0, 20).map((run) => (
                        <tr key={run.run_id}>
                          <td>{formatTs(run.recorded_at)}</td>
                          <td>
                            <a href={run.link_url} target="_blank" rel="noreferrer">
                              {run.link_title || run.link_url}
                            </a>
                          </td>
                          <td>{run.query}</td>
                          <td>{run.added_facts}</td>
                          <td>{run.outcome}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </section>
        ) : (
          <section className="enrollCard worldTeachCard">
            <div className="enrollCardHeader">
              <h3>Progressive Workflow</h3>
            </div>
            <p className="hint">
              Pick a concept from Knowledge Library using <strong>Manage</strong> to open Step 1, Step 2, and Step 3 forms.
            </p>
            {concepts.length ? (
              <button className="enrollBizBtn enrollBtnPrimary" onClick={() => void loadConcept(concepts[0].concept_id)}>
                Open First Concept
              </button>
            ) : (
              <p className="hint">Create your first concept above to start progressive disclosure forms.</p>
            )}
          </section>
        )}
      </Motion.section>
    </main>
  );
}
