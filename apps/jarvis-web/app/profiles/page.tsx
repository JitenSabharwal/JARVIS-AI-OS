"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { BookText, Database, Globe2, RefreshCw, UserCircle2, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  enrichWorldConcept,
  enrichWorldConceptWithSource,
  getWorldConcept,
  getWorldConceptProfileGraph,
  listVisionIdentities,
  listWorldConcepts,
  teachWorldConcept,
  updateWorldConcept,
  type ProfileGraph,
  type VisionIdentity,
  type WorldConcept
} from "../../lib/api";

type WorldEnrichSource = "web" | "linkedin" | "github" | "google";

function normalize(v: string): string {
  return String(v || "").trim().toLowerCase();
}

function findConceptForIdentity(identity: VisionIdentity | null, concepts: WorldConcept[]): WorldConcept | null {
  if (!identity) return null;
  const pid = normalize(identity.person_id);
  const name = normalize(identity.display_name);
  if (!pid && !name) return null;
  const byPersonId = concepts.find((c) => normalize(String(c?.metadata?.person_id || "")) === pid);
  if (byPersonId) return byPersonId;
  const byIdentityName = concepts.find((c) => normalize(String(c?.metadata?.identity_name || "")) === name);
  if (byIdentityName) return byIdentityName;
  const byTopic = concepts.find((c) => normalize(c.topic) === name);
  return byTopic || null;
}

function defaultQueryFor(source: WorldEnrichSource, displayName: string): string {
  const who = String(displayName || "").trim() || "person";
  if (source === "linkedin") return who;
  if (source === "github") return `${who} github profile`;
  if (source === "google") return `${who} professional profile`;
  return who;
}

export default function ProfilesPage() {
  const Motion = motion as any;
  const [identities, setIdentities] = useState<VisionIdentity[]>([]);
  const [concepts, setConcepts] = useState<WorldConcept[]>([]);
  const [selectedPersonId, setSelectedPersonId] = useState("");
  const [selectedConcept, setSelectedConcept] = useState<WorldConcept | null>(null);
  const [profileGraph, setProfileGraph] = useState<ProfileGraph | null>(null);
  const [graphBusy, setGraphBusy] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("profiles: loading...");
  const [enrichSource, setEnrichSource] = useState<WorldEnrichSource>("google");
  const [enrichQuery, setEnrichQuery] = useState("");
  const [queryTouched, setQueryTouched] = useState(false);
  const [selectedElementKeys, setSelectedElementKeys] = useState<Record<string, boolean>>({});

  const selectedIdentity = useMemo(
    () => identities.find((x) => x.person_id === selectedPersonId) || null,
    [identities, selectedPersonId]
  );

  const conceptMap = useMemo(() => {
    const map = new Map<string, WorldConcept | null>();
    for (const id of identities) {
      map.set(id.person_id, findConceptForIdentity(id, concepts));
    }
    return map;
  }, [identities, concepts]);

  useEffect(() => {
    void refreshAll();
  }, []);

  useEffect(() => {
    if (!selectedIdentity) return;
    if (queryTouched) return;
    setEnrichQuery(defaultQueryFor(enrichSource, selectedIdentity.display_name));
  }, [selectedIdentity, enrichSource, queryTouched]);

  useEffect(() => {
    const concept = findConceptForIdentity(selectedIdentity, concepts);
    if (!concept) {
      setSelectedConcept(null);
      setProfileGraph(null);
      setSelectedElementKeys({});
      return;
    }
    void loadConceptDetail(concept.concept_id);
  }, [selectedIdentity, concepts]);

  useEffect(() => {
    setSelectedElementKeys({});
  }, [selectedConcept?.concept_id]);

  async function refreshAll() {
    try {
      const [ids, rows] = await Promise.all([listVisionIdentities(), listWorldConcepts(200)]);
      setIdentities(ids);
      setConcepts(rows);
      if (!ids.length) {
        setSelectedPersonId("");
        setSelectedConcept(null);
        setStatus("profiles: no enrolled identities yet");
        return;
      }
      if (!ids.some((x) => x.person_id === selectedPersonId)) {
        setSelectedPersonId(ids[0].person_id);
      }
      setStatus(`profiles: ${ids.length} identities • ${rows.length} knowledge concepts`);
    } catch (err) {
      setStatus(err instanceof Error ? "profiles: " + err.message : "profiles: unavailable");
    }
  }

  async function loadConceptDetail(conceptId: string) {
    try {
      const row = await getWorldConcept(conceptId);
      setSelectedConcept(row);
      await loadProfileGraph(conceptId);
    } catch {
      const fallback = concepts.find((c) => c.concept_id === conceptId) || null;
      setSelectedConcept(fallback);
      if (fallback?.concept_id) {
        await loadProfileGraph(fallback.concept_id);
      } else {
        setProfileGraph(null);
      }
    }
  }

  async function loadProfileGraph(conceptId: string) {
    setGraphBusy(true);
    try {
      const graph = await getWorldConceptProfileGraph(conceptId);
      setProfileGraph(graph);
    } catch {
      setProfileGraph(null);
    } finally {
      setGraphBusy(false);
    }
  }

  async function ensureConceptForSelected(): Promise<string | null> {
    if (!selectedIdentity) return null;
    const existing = conceptMap.get(selectedIdentity.person_id) || null;
    if (existing?.concept_id) return existing.concept_id;
    const learned = await teachWorldConcept({
      topic: selectedIdentity.display_name,
      notes: `Profile manager initialized concept for ${selectedIdentity.display_name}.`,
      tags: ["profile", "enrollment", "identity"],
      metadata: {
        source: "profiles_manager",
        person_id: selectedIdentity.person_id,
        identity_name: selectedIdentity.display_name
      },
      enrich_web: false
    });
    await refreshAll();
    return learned.concept_id;
  }

  async function runProfileEnrichment() {
    if (!selectedIdentity) return;
    setBusy(true);
    try {
      const conceptId = await ensureConceptForSelected();
      if (!conceptId) {
        setStatus("profiles: failed to resolve profile concept");
        return;
      }
      const query = String(enrichQuery || "").trim() || defaultQueryFor(enrichSource, selectedIdentity.display_name);
      setStatus(`profiles: enriching ${selectedIdentity.display_name} via ${enrichSource}...`);
      if (enrichSource === "web") {
        await enrichWorldConcept(conceptId, 6);
      } else {
        await enrichWorldConceptWithSource(conceptId, {
          target_source: enrichSource,
          query,
          user_id: "web_user",
          max_items: 6,
          run_adapters: true
        });
      }
      await refreshAll();
      const latest = findConceptForIdentity(selectedIdentity, concepts);
      if (latest?.concept_id) await loadConceptDetail(latest.concept_id);
      setStatus(`profiles: enriched ${selectedIdentity.display_name}`);
    } catch (err) {
      setStatus(err instanceof Error ? "profiles: " + err.message : "profiles: enrichment failed");
    } finally {
      setBusy(false);
    }
  }

  function extractionElementsForSelected(): Array<{ key: string; kind: string; selector: string; text: string }> {
    const md = (selectedConcept?.metadata || {}) as Record<string, unknown>;
    const rows = Array.isArray(md.linkedin_extraction_elements) ? md.linkedin_extraction_elements : [];
    const out: Array<{ key: string; kind: string; selector: string; text: string }> = [];
    for (const row of rows) {
      if (!row || typeof row !== "object") continue;
      const r = row as Record<string, unknown>;
      const kind = String(r.kind || "other").trim().toLowerCase();
      const selector = String(r.selector || "").trim();
      const text = String(r.text || "").trim();
      if (!text) continue;
      const key = `${kind}|${selector}|${text.slice(0, 80)}`;
      out.push({ key, kind, selector, text });
      if (out.length >= 220) break;
    }
    return out;
  }

  async function applySelectedElementsToProfileMap() {
    if (!selectedConcept) return;
    const md = (selectedConcept.metadata || {}) as Record<string, unknown>;
    const rows = extractionElementsForSelected().filter((r) => selectedElementKeys[r.key]);
    if (!rows.length) {
      setStatus("profiles: select at least one extracted element");
      return;
    }
    const existingMap = (md.linkedin_profile_map || {}) as Record<string, unknown>;
    const contacts = new Set<string>(Array.isArray(existingMap.contacts) ? (existingMap.contacts as string[]) : []);
    const skills = new Set<string>(Array.isArray(existingMap.skills) ? (existingMap.skills as string[]) : []);
    const experience = new Set<string>(Array.isArray(existingMap.experience) ? (existingMap.experience as string[]) : []);
    const companies = new Set<string>(Array.isArray(existingMap.companies) ? (existingMap.companies as string[]) : []);
    const education = new Set<string>(Array.isArray(existingMap.education) ? (existingMap.education as string[]) : []);
    const locations = new Set<string>(Array.isArray(existingMap.locations) ? (existingMap.locations as string[]) : []);

    for (const row of rows) {
      const text = row.text.replace(/\s+/g, " ").trim();
      if (!text) continue;
      if (row.kind === "contact") {
        contacts.add(text);
      } else if (row.kind === "skills") {
        for (const m of text.match(/\b(Python|Java|Golang|TypeScript|JavaScript|React|Node\.js|Kubernetes|AWS|Azure|GCP|Machine Learning|AI|DevOps|SQL|Docker|Terraform)\b/gi) || []) {
          skills.add(m.trim());
        }
      } else if (row.kind === "experience") {
        experience.add(text);
        const at = text.match(/\b([A-Za-z][A-Za-z0-9&.,'() \-]{2,90})\s+at\s+([A-Z][A-Za-z0-9&.,'() \-]{2,90})\b/);
        if (at) companies.add(at[2].trim());
      } else if (row.kind === "education") {
        education.add(text);
      } else {
        const loc = text.match(/\b([A-Z][A-Za-z.'\- ]{1,60}(?:Metropolitan Area| Area|, [A-Z][A-Za-z.'\- ]{1,60}))\b/);
        if (loc) locations.add(loc[1].trim());
      }
    }

    const mergedMap = {
      ...existingMap,
      contacts: Array.from(contacts).slice(0, 12),
      skills: Array.from(skills).slice(0, 20),
      experience: Array.from(experience).slice(0, 12),
      companies: Array.from(companies).slice(0, 8),
      education: Array.from(education).slice(0, 8),
      locations: Array.from(locations).slice(0, 8)
    };
    try {
      setBusy(true);
      setStatus("profiles: applying selected elements...");
      await updateWorldConcept(selectedConcept.concept_id, {
        metadata: {
          linkedin_profile_map: mergedMap,
          linkedin_manual_selected_keys: rows.map((r) => r.key),
          linkedin_manual_selected_at: Date.now() / 1000
        }
      });
      await loadConceptDetail(selectedConcept.concept_id);
      setStatus("profiles: applied selected extracted elements");
    } catch (err) {
      setStatus(err instanceof Error ? "profiles: " + err.message : "profiles: failed to apply selected elements");
    } finally {
      setBusy(false);
    }
  }

  const stats = useMemo(() => {
    const c = selectedConcept;
    if (!c) return { notes: 0, facts: 0, links: 0, runs: 0 };
    return {
      notes: Number(c.notes_count || 0),
      facts: Number(c.web_facts_count || 0),
      links: Number(c.reference_links_count || 0),
      runs: Number(c.learning_runs_count || 0)
    };
  }, [selectedConcept]);

  return (
    <main className="enrollShell enrollBusiness profileShell">
      <Motion.section
        className="enrollMain enrollFrame"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: "easeOut" }}
      >
        <div className="enrollHeader enrollTopBar">
          <div className="enrollTitleBlock">
            <p className="enrollEyebrow">
              <UserCircle2 size={14} />
              Profile Intelligence
            </p>
            <h1>Enrolled Profile Manager</h1>
            <p>See everything Jarvis has learned for each enrolled identity and manage enrichment per profile.</p>
          </div>
          <div className="controlsGrid">
            <Link href="/" className="enrollBizBtn enrollBtnGhost">
              Live Console
            </Link>
            <Link href="/enroll" className="enrollBizBtn enrollBtnGhost">
              Enrollment Studio
            </Link>
            <Link href="/world-teaching" className="enrollBizBtn enrollBtnGhost">
              World Teaching
            </Link>
          </div>
        </div>

        <div className="enrollStatsRow">
          <article className="enrollStatCard">
            <span className="enrollStatLabel">Enrolled Profiles</span>
            <strong className="enrollStatValue">{identities.length}</strong>
          </article>
          <article className="enrollStatCard">
            <span className="enrollStatLabel">Mapped Concepts</span>
            <strong className="enrollStatValue">{Array.from(conceptMap.values()).filter(Boolean).length}</strong>
          </article>
          <article className="enrollStatCard">
            <span className="enrollStatLabel">Known Facts (Selected)</span>
            <strong className="enrollStatValue">{stats.facts}</strong>
          </article>
        </div>

        <p className="hint">{status}</p>

        <div className="profileLayout">
          <section className="enrollCard">
            <div className="enrollCardHeader">
              <h3>Profiles</h3>
              <button className="enrollBizBtn enrollBtnGhost" disabled={busy} onClick={() => void refreshAll()}>
                <RefreshCw size={14} />
                Refresh
              </button>
            </div>
            {identities.length ? (
              <div className="enrollIdentityList">
                {identities.map((id) => {
                  const mapped = conceptMap.get(id.person_id) || null;
                  const active = selectedPersonId === id.person_id;
                  return (
                    <button
                      key={id.person_id}
                      className={"profileItemBtn" + (active ? " active" : "")}
                      onClick={() => setSelectedPersonId(id.person_id)}
                    >
                      <div className="profileItemTop">
                        <strong>{id.display_name}</strong>
                        <span>{id.sample_count} samples</span>
                      </div>
                      <span className="profileItemMeta">{mapped ? "knowledge linked" : "no concept mapped"}</span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <p className="hint">No enrolled profiles yet. Enroll from Enrollment Studio first.</p>
            )}
          </section>

          <section className="enrollCard">
            <div className="enrollCardHeader">
              <h3>Profile Knowledge</h3>
              <span className="enrollBadge">{selectedIdentity?.display_name || "Select a profile"}</span>
            </div>
            {selectedIdentity ? (
              <>
                <div className="profileQuickStats">
                  <span className="enrollBadge">
                    <BookText size={12} />
                    Notes: {stats.notes}
                  </span>
                  <span className="enrollBadge">
                    <Globe2 size={12} />
                    Web facts: {stats.facts}
                  </span>
                  <span className="enrollBadge">
                    <Database size={12} />
                    Links: {stats.links}
                  </span>
                  <span className="enrollBadge">
                    <Zap size={12} />
                    Runs: {stats.runs}
                  </span>
                </div>

                <div className="liveWorldTeachRow" style={{ marginTop: 8 }}>
                  <select
                    className="liveWorldSourceSelect"
                    value={enrichSource}
                    onChange={(e) => {
                      setEnrichSource(e.target.value as WorldEnrichSource);
                      setQueryTouched(false);
                    }}
                  >
                    <option value="google">Google Search MCP</option>
                    <option value="github">GitHub MCP</option>
                    <option value="linkedin">LinkedIn MCP</option>
                    <option value="web">Web Enrich (default)</option>
                  </select>
                  <input
                    value={enrichQuery}
                    onChange={(e) => {
                      setQueryTouched(true);
                      setEnrichQuery(e.target.value);
                    }}
                    placeholder="Enrichment query for selected profile"
                  />
                  <button className="enrollBizBtn enrollBtnPrimary" disabled={busy} onClick={() => void runProfileEnrichment()}>
                    {busy ? "Running..." : "Run Enrichment"}
                  </button>
                </div>

                {selectedConcept ? (
                  <div className="profileKnowledgeBody">
                    <p className="hint">
                      Concept: <strong>{selectedConcept.topic}</strong> • Updated:{" "}
                      {new Date((selectedConcept.updated_at || 0) * 1000).toLocaleString()}
                    </p>
                    {selectedConcept.latest_note ? (
                      <div className="profilePanel">
                        <strong>Latest Note</strong>
                        <p>{selectedConcept.latest_note}</p>
                      </div>
                    ) : null}
                    {selectedConcept.web_facts?.length ? (
                      <div className="profilePanel">
                        <strong>Web Facts</strong>
                        <div className="profileList">
                          {selectedConcept.web_facts.slice(0, 12).map((fact, idx) => (
                            <a key={`${fact.url}-${idx}`} href={fact.url} target="_blank" rel="noreferrer" className="profileLinkRow">
                              <span>{fact.title || fact.url}</span>
                              <small>{fact.source_type}</small>
                            </a>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <p className="hint">No web facts yet.</p>
                    )}
                    {(() => {
                      const md = (selectedConcept.metadata || {}) as Record<string, unknown>;
                      const snapshot = String(md.linkedin_snapshot_data_url || "").trim();
                      const elements = extractionElementsForSelected();
                      return (
                        <div className="profilePanel">
                          <div className="profileGraphHeader">
                            <strong>Snapshot + Element Mapping</strong>
                            <span className="enrollBadge">{elements.length} found</span>
                          </div>
                          {snapshot ? (
                            <div style={{ borderRadius: 12, overflow: "hidden", border: "1px solid rgba(120,170,220,0.25)" }}>
                              <img src={snapshot} alt="LinkedIn extraction snapshot" style={{ width: "100%", display: "block" }} />
                            </div>
                          ) : (
                            <p className="hint">
                              No snapshot yet. Run LinkedIn MCP enrichment for this profile, then refresh this page.
                            </p>
                          )}
                          {elements.length ? (
                            <>
                              <p className="hint">Mark missing skills, experience, contact, or education blocks then apply.</p>
                              <div className="profileList">
                                {elements.slice(0, 120).map((el) => (
                                  <label key={el.key} className="profileEdgeRow" style={{ cursor: "pointer" }}>
                                    <div className="profileEdgeMain">
                                      <input
                                        type="checkbox"
                                        checked={Boolean(selectedElementKeys[el.key])}
                                        onChange={(e) =>
                                          setSelectedElementKeys((prev) => ({
                                            ...prev,
                                            [el.key]: e.target.checked
                                          }))
                                        }
                                      />
                                      <span className="profileEdgeRelation">{el.kind}</span>
                                      <strong>{el.text}</strong>
                                    </div>
                                    <small>{el.selector || "div"}</small>
                                  </label>
                                ))}
                              </div>
                              <button
                                className="enrollBizBtn enrollBtnPrimary"
                                disabled={busy}
                                onClick={() => void applySelectedElementsToProfileMap()}
                              >
                                {busy ? "Applying..." : "Apply Selected Elements"}
                              </button>
                            </>
                          ) : (
                            <p className="hint">
                              No extracted elements yet. After enrichment, this section will list detected page blocks.
                            </p>
                          )}
                        </div>
                      );
                    })()}
                    {selectedConcept.reference_links?.length ? (
                      <div className="profilePanel">
                        <strong>Reference Links</strong>
                        <div className="profileList">
                          {selectedConcept.reference_links.slice(0, 12).map((link) => (
                            <a key={link.link_id} href={link.url} target="_blank" rel="noreferrer" className="profileLinkRow">
                              <span>{link.title || link.url}</span>
                              <small>{link.source_type}</small>
                            </a>
                          ))}
                        </div>
                      </div>
                    ) : null}
                    {selectedConcept.learning_runs?.length ? (
                      <div className="profilePanel">
                        <strong>Deep Search Runs</strong>
                        <div className="profileList">
                          {selectedConcept.learning_runs.slice(0, 20).map((run) => (
                            <div key={run.run_id} className="profileEdgeRow">
                              <div className="profileEdgeMain">
                                <span className="profileEdgeRelation">{run.mode || "run"}</span>
                                <strong>{run.query || run.link_title || run.link_url || "enrichment run"}</strong>
                              </div>
                              <small>
                                {run.outcome || "unknown"} • facts {Number(run.added_facts || 0)} • results{" "}
                                {Number(run.query_result_count || 0)} • {new Date((run.recorded_at || 0) * 1000).toLocaleString()}
                              </small>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <p className="hint">No deep search runs yet.</p>
                    )}
                    <div className="profilePanel">
                      <div className="profileGraphHeader">
                        <strong>Profile Relation Graph</strong>
                        <span className="enrollBadge">
                          {graphBusy
                            ? "loading..."
                            : profileGraph
                              ? `${profileGraph.edges.length} relations • ${profileGraph.nodes.length} nodes`
                              : "not available"}
                        </span>
                      </div>
                      {profileGraph?.edges?.length ? (
                        <div className="profileList">
                          {profileGraph.edges.slice(0, 24).map((edge, idx) => {
                            const to = (profileGraph.nodes || []).find((n) => n.id === edge.to);
                            return (
                              <div key={`${edge.from}-${edge.to}-${edge.relation}-${idx}`} className="profileEdgeRow">
                                <div className="profileEdgeMain">
                                  <span className="profileEdgeRelation">{edge.relation}</span>
                                  <strong>{to?.value || edge.to}</strong>
                                </div>
                                <small>
                                  {edge.source || "profile"}
                                  {typeof edge.confidence === "number" ? ` • ${Math.round(edge.confidence * 100)}%` : ""}
                                </small>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="hint">
                          {profileGraph?.reason
                            ? `Graph fallback active: ${profileGraph.reason}`
                            : "No mapped relations yet. Run enrichment and refresh."}
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <p className="hint">No knowledge concept mapped yet. Run enrichment to auto-create and populate one.</p>
                )}
              </>
            ) : (
              <p className="hint">Select an enrolled profile from the left panel.</p>
            )}
          </section>
        </div>
      </Motion.section>
    </main>
  );
}
