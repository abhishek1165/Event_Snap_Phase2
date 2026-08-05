// ════════════════════════════════════════════════════════════════════
// Event Snap — Design Tokens (single source of truth)
// ════════════════════════════════════════════════════════════════════
// Reuses the identity the project was built on (ink / paper / violet /
// mint + Instrument Sans / Inter / IBM Plex Mono) and extends it with
// semantic roles, gradients, glows and motion primitives so every page
// speaks the same visual language.
//
// The raw hex values here are mirrored as CSS variables in index.css so
// the Tailwind app pages can use bg-surface / text-accent instead of
// hardcoding hex. Landing sections import the `T` object directly.

export const T = {
  // ── Surfaces (cinematic "elevated dark", layered for depth) ──
  ink:       "#08090d",   // page background (near-black, slight cool)
  ink2:      "#0b0c12",   // hero / feature band
  paper:     "#0e1016",   // card background
  paper2:    "#141722",   // elevated / hover card
  paper3:    "#1a1d29",   // raised inputs, active tabs

  // ── Lines ──
  frame:     "#262b38",   // standard border
  frameSoft: "rgba(255,255,255,.08)", // hairline border

  // ── Text ──
  text:      "#f1efe9",   // primary (warm off-white)
  dim:       "#9a9dae",   // secondary
  faint:     "#5c5f70",   // tertiary / mono labels

  // ── Brand / accent (iris violet = AI / technology / brand) ──
  signal:    "#6d5ef5",   // primary violet
  signal2:   "#8b7bff",   // lighter violet (highlights, glows)
  signalSoft:"rgba(109,94,245,.16)",

  // ── Semantic: MATCH (mint = face-match success ONLY) ──
  conf:      "#34d399",
  confSoft:  "rgba(52,211,153,.14)",

  // ── Semantic: MEMORY (amber = human / event names, used sparingly) ──
  amber:     "#e8b04b",
  amberSoft: "rgba(232,176,75,.14)",

  // convenience gradient strings (kept on T for inline-style contexts)
  heroText: "linear-gradient(100deg,#8b7bff 0%,#34d399 90%)",
  brandGradient: "linear-gradient(135deg,#6d5ef5 0%,#8b7bff 100%)",
};

// ── Type ────────────────────────────────────────────────────────────
export const DISPLAY = "'Instrument Sans', system-ui, sans-serif"; // headings
export const BODY    = "'Inter', system-ui, sans-serif";           // body
export const MONO    = "'IBM Plex Mono', monospace";               // labels / data

// Heading style object — tight tracking, display face
export const R = { fontFamily: DISPLAY, fontWeight: 600, letterSpacing: "-.02em" };

// Body style object
export const B = { fontFamily: BODY };

// ── Gradients ───────────────────────────────────────────────────────
export const GRAD = {
  brand:   "linear-gradient(135deg,#6d5ef5 0%,#8b7bff 100%)",   // primary CTA
  brandSoft:"linear-gradient(135deg,rgba(109,94,245,.18),rgba(139,123,255,.05))",
  match:   "linear-gradient(135deg,#34d399 0%,#10b981 100%)",   // success
  heroText:"linear-gradient(100deg,#8b7bff 0%,#34d399 90%)",    // hero headline span
  mesh:    "radial-gradient(60% 80% at 50% 0%,rgba(109,94,245,.22),transparent 70%),radial-gradient(50% 60% at 85% 100%,rgba(52,211,153,.12),transparent 70%)",
  edge:    "linear-gradient(180deg,rgba(255,255,255,.12) 0%,rgba(255,255,255,.02) 100%)", // 1px luminous top-edge border
};

// ── Shadows (TINTED to the cool background hue, never harsh pure-black) ──
// Per impeccable §2: tint shadows to the background; Linear-style diffused.
export const SHADOW = {
  // restrained brand lift — subtle violet edge, NOT a neon glow (slop tell)
  brand: "0 1px 0 0 rgba(255,255,255,.05) inset, 0 10px 30px -12px rgba(109,94,245,.40)",
  match: "0 1px 0 0 rgba(255,255,255,.05) inset, 0 10px 30px -12px rgba(52,211,153,.35)",
  // layered card depth, cool-tinted
  card:  "0 1px 0 0 rgba(255,255,255,.04) inset, 0 24px 48px -28px rgba(8,9,20,.85)",
  lift:  "0 1px 0 0 rgba(255,255,255,.04) inset, 0 30px 60px -28px rgba(8,9,20,.9)",
  none:  "none",
};

// ── Motion ──────────────────────────────────────────────────────────
// Spring physics for interactions (emil), crisp expo-out for entrances.
// Exit ~60-70% of enter (Vercel motion rule).
export const EASE = {
  out:   [0.16, 1, 0.3, 1],    // expo-out — premium "settle"
  inOut: [0.65, 0, 0.35, 1],
  // emil-recommended spring for tactile UI
  spring: { type: "spring", stiffness: 320, damping: 30 },
  springSoft: { type: "spring", stiffness: 180, damping: 22 },
};

export const DUR = {
  micro: 0.18,   // hover / press feedback (150–300ms band)
  fast:  0.28,
  base:  0.45,
  slow:  0.7,
};

export const STAGGER = 0.06; // 60ms — within the 30–50ms+ recommendation

// ── Spacing (8dp rhythm) + radius ───────────────────────────────────
export const SPACE = { xs: 4, sm: 8, md: 16, lg: 24, xl: 40, xxl: 64 };
export const RADIUS = { sm: 8, md: 12, lg: 16, xl: 20, pill: 999 };

export default T;
