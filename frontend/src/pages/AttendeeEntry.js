import React, { useState, useRef, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { toast } from "sonner";
import api from "@/utils/api";
import { T, MONO, DISPLAY, R } from "@/design/tokens";
import { ApertureMark as LogoMark } from "@/components/Logo";
import { Eyebrow, Btn } from "@/components/landing/shared";

/* ── Responsive width hook ── */
function useW() {
  const [w, setW] = useState(() => window.innerWidth);
  useEffect(() => {
    const fn = () => setW(window.innerWidth);
    window.addEventListener("resize", fn, { passive: true });
    return () => window.removeEventListener("resize", fn);
  }, []);
  return w;
}

/* ── Stepper ── */
const STEP_LABELS = ["Code"];
function Stepper({ step }) {
  const w = useW();
  const small = w < 480;
  return (
    <div style={{ display:"flex", alignItems:"center", gap: small ? 6 : 10 }}>
      {STEP_LABELS.map((label, i) => {
        const n = i + 1;
        const isActive   = n === step;
        const isComplete = n < step;
        return (
          <React.Fragment key={n}>
            {i > 0 && <div style={{ width: small ? 14 : 30, height:1, background:T.frame }} />}
            <div style={{ display:"flex", alignItems:"center", gap:6, fontFamily:MONO, fontSize:".68rem", color: isActive ? T.text : T.faint, textTransform:"uppercase", letterSpacing:".05em" }}>
              <span style={{
                width:22, height:22, borderRadius:"50%", display:"flex", alignItems:"center", justifyContent:"center",
                fontFamily:"'Inter',system-ui,sans-serif", fontSize:".7rem", flexShrink:0,
                border:`1px solid ${isActive ? T.signal2 : isComplete ? T.conf : T.frame}`,
                color: isActive ? T.signal2 : isComplete ? "#04150e" : T.faint,
                background: isComplete ? T.conf : "transparent",
                boxShadow: isActive ? `0 0 0 3px rgba(139,123,255,.15)` : "none",
                transition:"all .3s ease",
              }}>
                {isComplete ? "✓" : n}
              </span>
              {!small && <em style={{ fontStyle:"normal" }}>{label}</em>}
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

/* ════════════════════════════════════════════════════
   STEP 1 — Event Code
════════════════════════════════════════════════════ */
function StepCode({ onNext }) {
  const w = useW();
  const [chars, setChars] = useState(Array(8).fill(""));
  const [loading, setLoading] = useState(false);
  const refs = useRef([]);

  const full = chars.every(c => c.length === 1);
  const codeStr = chars.join("").toUpperCase();

  /* box sizing responsive */
  const boxW = w < 380 ? 34 : w < 480 ? 38 : w < 600 ? 44 : 48;
  const boxH = w < 380 ? 46 : w < 480 ? 52 : w < 600 ? 56 : 60;
  const boxGap = w < 380 ? 5 : w < 480 ? 6 : 8;

  const handleInput = (i, val) => {
    const ch = val.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(-1);
    const next = [...chars]; next[i] = ch; setChars(next);
    if (ch && refs.current[i + 1]) refs.current[i + 1].focus();
  };

  const handleKeyDown = (i, e) => {
    if (e.key === "Backspace" && !chars[i] && refs.current[i - 1]) refs.current[i - 1].focus();
  };

  const handlePaste = (e) => {
    e.preventDefault();
    const text = (e.clipboardData.getData("text") || "").toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8);
    const next = Array(8).fill("");
    text.split("").forEach((c, idx) => { if (idx < 8) next[idx] = c; });
    setChars(next);
    const last = Math.min(text.length, 8) - 1;
    if (refs.current[last]) refs.current[last].focus();
  };

  const handleFind = async () => {
    if (!full || loading) return;
    setLoading(true);
    try {
      const res = await api.get(`/events/code/${codeStr}`);
      const event = res.data;
      if (event.status === "processing") { toast.error("Event is still processing. Please try again later."); setLoading(false); return; }
      if (event.status === "active" && event.total_photos === 0) { toast.error("This event has no photos yet."); setLoading(false); return; }
      if (event.faces_detected === 0) toast.warning("No faces detected yet. You can still try searching.");
      toast.success(`Found event: ${event.title}`);
      // Navigate directly to selfie search page
      onNext(event);
    } catch (err) {
      const errorMsg = err.response?.data?.detail || "Event not found";
      toast.error(errorMsg);
      setLoading(false);
    }
  };

  return (
    <div style={{ width:"100%", maxWidth:480, textAlign:"center", padding: w < 480 ? "0 4px" : 0 }}>
      <Eyebrow style={{ marginBottom: 16 }}>Find your photos</Eyebrow>
      <h1 style={{ ...R, fontSize:"clamp(1.7rem,5vw,2.4rem)", marginBottom:10 }}>What's the event code?</h1>
      <p style={{ color:T.dim, fontSize: w < 480 ? ".9rem" : ".98rem", lineHeight:1.6 }}>
        Ask your organizer for the eight-character code. It's the only thing standing between you and your photos.
      </p>

      {/* Code boxes — grouped with an accessible label */}
      <div role="group" aria-label="Event code. Enter each character" style={{ display:"flex", gap:boxGap, justifyContent:"center", margin:"28px 0 24px", flexWrap:"nowrap" }}>
        {chars.map((c, i) => (
          <input key={i}
            ref={el => refs.current[i] = el}
            value={c} maxLength={1}
            inputMode="text"
            aria-label={`Character ${i + 1} of 8`}
            autoComplete="off"
            onInput={e => handleInput(i, e.target.value)}
            onKeyDown={e => handleKeyDown(i, e)}
            onPaste={handlePaste}
            style={{
              width:boxW, height:boxH, textAlign:"center",
              fontFamily:MONO, fontSize: w < 480 ? "1rem" : "1.3rem", fontWeight:600,
              background:T.paper, border:`1.5px solid ${c ? T.signal2 : T.frame}`,
              borderRadius:10, color:T.text, textTransform:"uppercase",
              transition:"border-color .2s,box-shadow .2s",
              boxShadow: c ? `0 0 0 3px rgba(139,123,255,.12)` : "none",
              flexShrink:0,
            }}
          />
        ))}
      </div>

      <Btn solid onClick={handleFind} disabled={!full || loading} aria-label={loading ? "Checking event code, please wait" : "Find my photos"} style={{ width:"100%" }}>
        {loading ? (
          <><span role="status" aria-label="Loading" style={{ width:16, height:16, border:"2px solid rgba(255,255,255,.3)", borderTopColor:"#fff", borderRadius:"50%", display:"inline-block", animation:"lspin 0.7s linear infinite" }} /> Checking code…</>
        ) : "Find my photos →"}
      </Btn>

      <span style={{ display:"block", marginTop:20, fontSize:".85rem", color:T.faint }}>
        Are you the organizer? <Link to="/auth" style={{ color:T.signal2 }}>Go to dashboard</Link>
      </span>
    </div>
  );
}

/* ════════════════════════════════════════════════════
   ROOT PAGE - Simplified (removed unnecessary steps)
════════════════════════════════════════════════════ */
export default function AttendeeEntry() {
  const w = useW();
  const navigate = useNavigate();

  const handleEventFound = (event) => {
    // Navigate directly to selfie search page
    navigate(`/attend/${event.id}/selfie`, { state: { event } });
  };

  const handleBack = () => navigate("/");

  /* Header padding responsive */
  const hPad = w < 480 ? "18px 16px" : w < 720 ? "22px 24px" : "26px 40px";
  /* main padding */
  const mPad = w < 480 ? "16px 16px 80px" : "20px 24px 90px";

  return (
    <div style={{ minHeight:"100vh", display:"flex", flexDirection:"column", position:"relative", background:T.ink, color:T.text, fontFamily:"'Inter',system-ui,sans-serif", lineHeight:1.6, WebkitFontSmoothing:"antialiased", overflowX:"hidden" }}>
      <style>{`
        @keyframes lspin { to { transform:rotate(360deg); } }
        * { box-sizing: border-box; }
        input { -webkit-appearance: none; }
        /* ── Visually hidden (screen-reader only) ── */
        .sr-only {
          position:absolute; width:1px; height:1px;
          padding:0; margin:-1px; overflow:hidden;
          clip:rect(0,0,0,0); white-space:nowrap; border:0;
        }
        /* ── Focus ring for inputs & buttons ── */
        :focus-visible { outline: 2px solid #8b7bff; outline-offset: 3px; border-radius: 4px; }
        /* ── Reduced motion ── */
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
          }
        }
      `}</style>

      {/* Ambient glow — smaller on mobile */}
      <div style={{ position:"fixed", width: w < 480 ? 360 : 640, height: w < 480 ? 360 : 640, top: w < 480 ? -180 : -320, left:"50%", transform:"translateX(-50%)", background:"radial-gradient(circle,rgba(109,94,245,.14),transparent 65%)", pointerEvents:"none", zIndex:0 }} />

      {/* ── Skip link (WCAG 2.4.1) ── */}
      <a href="#ls-flow-main" className="sr-only" style={{ position:"absolute", top:8, left:8, zIndex:100, background:T.signal, color:"#fff", padding:"8px 14px", borderRadius:8, fontWeight:600, fontSize:".85rem", textDecoration:"none" }}
        onFocus={e => e.currentTarget.style.clip = "auto"}
        onBlur={e => e.currentTarget.style.clip = "rect(0,0,0,0)"}>
        Skip to main content
      </a>

      {/* ── Header ── */}
      <header style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:hPad, position:"relative", zIndex:5, gap:8 }}>
        {/* Back button */}
        <button onClick={handleBack} aria-label="Back to home"
          style={{ fontSize: w < 480 ? ".82rem" : ".9rem", color:T.dim, background:"none", border:"none", cursor:"pointer", fontFamily:"inherit", transition:"color .2s", flexShrink:0, whiteSpace:"nowrap" }}
          onMouseEnter={e => e.currentTarget.style.color = T.text}
          onMouseLeave={e => e.currentTarget.style.color = T.dim}>
          ← {w < 400 ? "" : "Home"}
        </button>

        {/* Center logo — hidden on very small if stepper overflows */}
        {w >= 360 && (
          <div style={{ position:"absolute", left:"50%", transform:"translateX(-50%)", display:"flex", alignItems:"center", gap:8, ...R, fontSize: w < 480 ? ".9rem" : "1rem", pointerEvents:"none" }}>
            <LogoMark />
            {w >= 480 && "Event Snap"}
          </div>
        )}

        {/* Stepper */}
        <div style={{ flexShrink:0, marginLeft:"auto" }}>
          <Stepper step={1} />
        </div>
      </header>

      {/* ── Main ── */}
      <main id="ls-flow-main" style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:mPad, position:"relative", zIndex:2 }}>
        <div id="ls-flow-step" style={{ width:"100%", maxWidth: w < 600 ? "100%" : 540, display:"flex", flexDirection:"column", alignItems:"center" }}>
          <StepCode onNext={handleEventFound} />
        </div>
      </main>
    </div>
  );
}
