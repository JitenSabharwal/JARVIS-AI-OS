"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Bot, UserCircle2, Waves } from "lucide-react";

import type { ChatMessage } from "../lib/types";

type Props = {
  messages: ChatMessage[];
  live?: boolean;
  liveDraftText?: string;
  assistantDraftText?: string;
};

export function ChatWindow({ messages, live = false, liveDraftText = "", assistantDraftText = "" }: Props) {
  const Motion = motion as any;
  return (
    <Motion.section
      className="chatPane glass"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      <div className="chatHeader">
        <h3>Conversation</h3>
        <p>Realtime multimodal call view</p>
      </div>
      <div className="voiceStage">
        <p className="voiceStageLabel">
          <Waves size={13} /> Maia is listening...
        </p>
        <div className="voiceOrbWrap">
          <Motion.div
            className="voiceOrb"
            animate={
              live
                ? { scale: [1, 1.045, 1], opacity: [0.9, 1, 0.9] }
                : { scale: 1, opacity: 0.82 }
            }
            transition={live ? { duration: 2.2, repeat: Infinity, ease: "easeInOut" } : { duration: 0.2 }}
          />
        </div>
      </div>
      <div className="messageList">
        {messages.length === 0 ? (
          <div className="emptyChat">
            <h4>Start speaking to Jarvis</h4>
            <p>Use Start Call for continuous conversation and camera grounding.</p>
          </div>
        ) : null}
        <AnimatePresence initial={false}>
          {messages.map((m) => (
            <Motion.div
              key={m.id}
              className={"bubble " + m.role}
              initial={{ opacity: 0, y: 10, scale: 0.985 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <div className="bubbleMeta">
                <span className="role">
                  {m.role === "assistant" ? <Bot size={12} /> : <UserCircle2 size={12} />}
                  {m.role === "assistant" ? "Jarvis" : "You"}
                </span>
                <span className="msgTime">
                  {new Date(m.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </div>
              <p>{m.content}</p>
            </Motion.div>
          ))}
        </AnimatePresence>
        {liveDraftText.trim() ? (
          <Motion.div
            className="bubble user draft"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.15 }}
          >
            <div className="bubbleMeta">
              <span className="role">
                <UserCircle2 size={12} />
                You (live)
              </span>
              <span className="msgTime">typing…</span>
            </div>
            <p>{liveDraftText}</p>
          </Motion.div>
        ) : null}
        {assistantDraftText.trim() ? (
          <Motion.div
            className="bubble assistant draft"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.15 }}
          >
            <div className="bubbleMeta">
              <span className="role">
                <Bot size={12} />
                Jarvis (streaming)
              </span>
              <span className="msgTime">writing…</span>
            </div>
            <p>{assistantDraftText}</p>
          </Motion.div>
        ) : null}
      </div>
    </Motion.section>
  );
}
