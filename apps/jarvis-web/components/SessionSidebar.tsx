"use client";

import { motion } from "framer-motion";
import { Globe2, LayoutDashboard, Plus, Sparkles, UserRoundCog, UserSquare2 } from "lucide-react";
import Link from "next/link";

import type { SessionItem } from "../lib/types";

type Props = {
  sessions: SessionItem[];
  activeId: string;
  onCreate: () => void;
  onSelect: (id: string) => void;
};

export function SessionSidebar({ sessions, activeId, onCreate, onSelect }: Props) {
  const Motion = motion as any;
  return (
    <Motion.aside
      className="sidebar glass"
      initial={{ opacity: 0, x: -14 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.34, ease: "easeOut" }}
    >
      <div className="sidebarTop">
        <div>
          <h2>Jarvis</h2>
          <p className="sidebarCaption">Live AI Console</p>
        </div>
      </div>
      <nav className="sidebarNav">
        <Link href="/" className="sidebarNavLink">
          <LayoutDashboard size={14} />
          Live Console
        </Link>
        <Link href="/world-teaching" className="sidebarNavLink">
          <Globe2 size={14} />
          World Teaching
        </Link>
        <Link href="/enroll" className="sidebarNavLink">
          <UserSquare2 size={14} />
          Enrollment Studio
        </Link>
        <Link href="/profiles" className="sidebarNavLink">
          <UserRoundCog size={14} />
          Profile Manager
        </Link>
      </nav>
      <div className="sidebarActions">
        <button className="btn btnWide" onClick={onCreate}>
          <Plus size={16} />
          New Session
        </button>
      </div>
      <div className="sidebarStatCard">
        <span className="statLabel">Active Sessions</span>
        <strong className="statValue">{sessions.length}</strong>
        <span className="statHint">Multi-window conversation memory</span>
      </div>
      <div className="sessionList">
        <p className="sessionListLabel">
          <Sparkles size={12} /> Recent
        </p>
        {sessions.map((s) => (
          <button
            key={s.id}
            className={"sessionItem" + (s.id === activeId ? " active" : "")}
            onClick={() => onSelect(s.id)}
          >
            <span className="sessionTitleRow">
              <span className="sessionTitle">{s.title}</span>
              <span className="sessionMeta">{s.messages.length} msg</span>
            </span>
            <span className="sessionPreview">
              {s.messages.length > 0 ? s.messages[s.messages.length - 1].content.slice(0, 56) : "No messages yet"}
            </span>
          </button>
        ))}
      </div>
    </Motion.aside>
  );
}
