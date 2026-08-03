// Shared design tokens & tiny atoms used across all landing sections
import React, { useState } from "react";

export const T = {
  ink:    "#08090d",
  paper:  "#0e1016",
  paper2: "#141722",
  frame:  "#262b38",
  frameSoft: "rgba(255,255,255,.08)",
  text:   "#f1efe9",
  dim:    "#9a9dae",
  faint:  "#5c5f70",
  signal: "#6d5ef5",
  signal2:"#8b7bff",
  conf:   "#34d399",
  confSoft:"rgba(52,211,153,.14)",
};

export const DISPLAY = "'Instrument Sans',system-ui,sans-serif";
export const MONO    = "'IBM Plex Mono',monospace";

export const R = { fontFamily: DISPLAY, fontWeight: 600, letterSpacing: "-.02em" };

export function LogoMark({ size = 30 }) {
  return (
    <div style={{ width: size, height: size, position: "relative", flexShrink: 0 }}>
      {[
        { top: 0,    left:  0, borderRight: "none", borderBottom: "none" },
        { top: 0,    right: 0, borderLeft:  "none", borderBottom: "none" },
        { bottom: 0, left:  0, borderRight: "none", borderTop:    "none" },
        { bottom: 0, right: 0, borderLeft:  "none", borderTop:    "none" },
      ].map((s, i) => (
        <span key={i} style={{ position: "absolute", width: 11, height: 11, border: `2px solid ${T.signal2}`, ...s }} />
      ))}
      <span style={{ position: "absolute", inset: size > 24 ? 9 : 7, borderRadius: "50%", background: T.conf }} />
    </div>
  );
}

export function Eyebrow({ children, style }) {
  return (
    <span style={{ fontFamily: MONO, fontSize: ".72rem", letterSpacing: ".14em", textTransform: "uppercase", color: T.signal2, display: "inline-flex", alignItems: "center", gap: 8, ...style }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: T.conf, boxShadow: `0 0 0 3px ${T.confSoft}`, flexShrink: 0 }} />
      {children}
    </span>
  );
}

export function Btn({ solid, children, onClick, style }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: "inline-flex", alignItems: "center", gap: 8,
        fontWeight: 600, fontSize: ".95rem",
        padding: "13px 22px", borderRadius: 11,
        border: solid ? "none" : `1px solid ${hov ? T.signal2 : T.frame}`,
        background: solid ? "linear-gradient(135deg,#6d5ef5,#8b7bff)" : "transparent",
        color: T.text, cursor: "pointer",
        boxShadow: solid ? (hov ? "0 12px 30px -6px rgba(109,94,245,.75)" : "0 8px 24px -8px rgba(109,94,245,.6)") : "none",
        transition: "transform .25s ease, box-shadow .25s ease, border-color .25s ease",
        transform: hov ? "translateY(-2px)" : "none",
        fontFamily: "inherit", whiteSpace: "nowrap",
        ...style,
      }}
    >
      {children}
    </button>
  );
}

export function Divider() {
  return <div className="lw"><div style={{ height: 1, background: T.frameSoft }} /></div>;
}
