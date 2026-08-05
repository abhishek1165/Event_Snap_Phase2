// ════════════════════════════════════════════════════════════════════
// Event Snap — shared landing tokens & atoms
// Re-exports the canonical design/tokens so every landing section speaks
// one language. API kept stable (T, R, MONO, DISPLAY, LogoMark, Eyebrow,
// Btn, Divider) so existing imports keep working.
// ════════════════════════════════════════════════════════════════════
import React, { useState } from "react";
import { T as TOKENS, R, DISPLAY, MONO, SHADOW } from "@/design/tokens";

export const T = TOKENS;
export { R, DISPLAY, MONO };

// One brand mark for the whole site (corner-bracket aperture).
// Landing sections import { LogoMark } and render <LogoMark /> expecting
// the bare aperture, so map LogoMark → ApertureMark.
export { ApertureMark as LogoMark } from "@/components/Logo";
export { default as Logo } from "@/components/Logo";

/**
 * Btn — landing CTA.
 * Refined per impeccable/design-taste skills: tinted shadow (not neon
 * glow), solid brand fill (gradient reserved for ONE hero moment),
 * tactile active scale, spring-free crisp easing.
 */
export function Btn({ solid, gradient, children, onClick, style, type = "button", "aria-label": ariaLabel, disabled }) {
  const [hov, setHov] = useState(false);
  const active = !disabled;
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 8,
        fontWeight: 600,
        fontSize: ".95rem",
        padding: "13px 22px",
        borderRadius: 11,
        border: solid || gradient ? "none" : `1px solid ${hov ? T.signal2 : T.frame}`,
        background: solid
          ? T.signal                    // solid brand violet (restrained)
          : gradient
          ? "linear-gradient(135deg,#6d5ef5,#8b7bff)" // hero-only gradient
          : "transparent",
        color: T.text,
        cursor: active ? "pointer" : "default",
        // tinted edge shadow, NOT neon glow (slop tell)
        boxShadow: solid || gradient
          ? hov
            ? SHADOW.brand
            : "0 1px 0 0 rgba(255,255,255,.16) inset, 0 6px 18px -10px rgba(109,94,245,.45)"
          : "none",
        transition: "transform .2s cubic-bezier(.16,1,.3,1), box-shadow .25s ease, border-color .25s ease",
        transform: active ? (hov ? "translateY(-2px)" : "none") : "none",
        fontFamily: "inherit",
        whiteSpace: "nowrap",
        opacity: active ? 1 : 0.5,
        ...style,
      }}
    >
      {children}
    </button>
  );
}

/**
 * Eyebrow — mono uppercase label. Use SPARINGLY (design-taste §4.7:
 * max 1 per 3 sections). No decorative dot prefix by default.
 */
export function Eyebrow({ children, style }) {
  return (
    <span
      style={{
        fontFamily: MONO,
        fontSize: ".72rem",
        letterSpacing: ".16em",
        textTransform: "uppercase",
        color: T.signal2,
        display: "inline-block",
        ...style,
      }}
    >
      {children}
    </span>
  );
}

export function Divider() {
  return (
    <div className="lw">
      <div style={{ height: 1, background: T.frameSoft }} />
    </div>
  );
}
