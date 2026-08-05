import React from "react";
import { T, DISPLAY, MONO } from "@/design/tokens";

/**
 * Event Snap — single brand logo used across every page.
 * An aperture mark: corner brackets (the viewfinder) + a blade dot
 * (the iris) in brand violet, with a mint match-dot at the centre.
 *
 * `variant`:
 *   "mark"     → aperture only
 *   "full"     → aperture + "Event Snap" wordmark
 *   "stacked"  → aperture above wordmark (auth/empty states)
 */
export function ApertureMark({ size = 30, animated = false }) {
  const bracket = Math.max(7, Math.round(size * 0.36));
  return (
    <div
      style={{
        width: size,
        height: size,
        position: "relative",
        flexShrink: 0,
      }}
      aria-hidden="true"
    >
      {/* four corner brackets = the viewfinder */}
      {[
        { top: 0, left: 0, borderRight: "none", borderBottom: "none", borderTopLeftRadius: 4 },
        { top: 0, right: 0, borderLeft: "none", borderBottom: "none", borderTopRightRadius: 4 },
        { bottom: 0, left: 0, borderRight: "none", borderTop: "none", borderBottomLeftRadius: 4 },
        { bottom: 0, right: 0, borderLeft: "none", borderTop: "none", borderBottomRightRadius: 4 },
      ].map((s, i) => (
        <span
          key={i}
          style={{
            position: "absolute",
            width: bracket,
            height: bracket,
            border: `2px solid ${T.signal2}`,
            ...s,
          }}
        />
      ))}
      {/* iris blade ring */}
      <span
        style={{
          position: "absolute",
          inset: size > 24 ? 8 : 6,
          borderRadius: "50%",
          border: `1.5px solid ${T.signal}`,
          opacity: 0.55,
        }}
      />
      {/* centre match dot */}
      <span
        style={{
          position: "absolute",
          inset: size > 24 ? 11 : 9,
          borderRadius: "50%",
          background: T.conf,
          boxShadow: `0 0 8px ${T.conf}`,
          animation: animated ? "es-pulse 2.4s ease-in-out infinite" : undefined,
        }}
      />
    </div>
  );
}

export default function Logo({
  size = 30,
  variant = "full",
  tagline = false,
  onClick,
  style,
}) {
  const wrap = (children) => (
    <span
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 11,
        cursor: onClick ? "pointer" : "default",
        textDecoration: "none",
        color: "inherit",
        fontFamily: "inherit",
        ...style,
      }}
    >
      {children}
    </span>
  );

  const wordmark = (
    <span style={{ display: "flex", flexDirection: "column", lineHeight: 1 }}>
      <span
        style={{
          fontFamily: DISPLAY,
          fontWeight: 700,
          fontSize: size > 24 ? "1.05rem" : ".95rem",
          letterSpacing: "-.02em",
          color: T.text,
        }}
      >
        Event&nbsp;Snap
      </span>
      {tagline && (
        <span
          style={{
            fontFamily: MONO,
            fontSize: ".6rem",
            letterSpacing: ".14em",
            textTransform: "uppercase",
            color: T.faint,
            marginTop: 2,
          }}
        >
          Face-matched photos
        </span>
      )}
    </span>
  );

  if (variant === "mark") return wrap(<ApertureMark size={size} />);
  if (variant === "stacked") {
    return (
      <span
        onClick={onClick}
        style={{
          display: "inline-flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 12,
          cursor: onClick ? "pointer" : "default",
          ...style,
        }}
      >
        <ApertureMark size={size} animated />
        {wordmark}
      </span>
    );
  }
  // full
  return wrap(
    <>
      <ApertureMark size={size} />
      {wordmark}
    </>
  );
}

/* keyframe for the animated centre dot (kept in JS so it ships wherever the logo does) */
if (typeof window !== "undefined" && !document.getElementById("es-logo-keyframes")) {
  const s = document.createElement("style");
  s.id = "es-logo-keyframes";
  s.textContent = "@keyframes es-pulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(.78);opacity:.7}}";
  document.head.appendChild(s);
}
