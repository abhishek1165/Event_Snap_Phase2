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
const STEP_LABELS = ["Code", "Verify", "Results"];
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
      onNext(event);
    } catch (err) {
      toast.error(err.response?.data?.detail || "Event not found");
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
   STEP 2 — Selfie / Upload
════════════════════════════════════════════════════ */
function StepVerify({ event, onNext }) {
  const w = useW();
  const [tab, setTab]       = useState("selfie");
  const [faceDetected, setFace] = useState(false);
  const [fileReady, setFile]    = useState(false);
  const [fileName, setFileName] = useState(null);
  const [uploadFile, setUploadFile] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (tab !== "selfie") return;
    setFace(false);
    const t = setTimeout(() => setFace(true), 1200);
    return () => clearTimeout(t);
  }, [tab]);

  const handleShutter = () => {
    if (!faceDetected) return;
    const canvas = document.createElement("canvas");
    canvas.width = 4; canvas.height = 4;
    canvas.toBlob(blob => { if (blob) onNext(event, blob); });
  };

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setUploadFile(f); setFileName(f.name); setFile(true);
  };

  const handleUploadContinue = () => {
    if (!fileReady || !uploadFile) return;
    onNext(event, uploadFile);
  };

  /* viewfinder aspect ratio: taller on small screens */
  const vfAspect = w < 480 ? "3/4" : "4/3";

  return (
    <div style={{ width:"100%", maxWidth: w < 600 ? "100%" : 460, textAlign:"center", padding: w < 480 ? "0 4px" : 0 }}>
      <Eyebrow style={{ marginBottom: 10 }}>{event?.title || "Event"} · verify it's you</Eyebrow>
      <h1 style={{ ...R, fontSize:"clamp(1.7rem,5vw,2.4rem)", marginBottom:10 }}>Show us your face.</h1>
      <p style={{ color:T.dim, fontSize: w < 480 ? ".9rem" : ".98rem", marginBottom:24, lineHeight:1.6 }}>
        One photo, nothing else. We turn it into a match key, never a photo we keep.
      </p>

      {/* Tabs — ARIA tablist pattern */}
      <div role="tablist" aria-label="Verification method" style={{ display:"flex", background:T.paper, border:`1px solid ${T.frameSoft}`, borderRadius:12, padding:4, marginBottom:22 }}>
        {[["selfie","Take Selfie"],["upload","Upload Photo"]].map(([key,label]) => (
          <button key={key}
            role="tab"
            aria-selected={tab === key}
            aria-controls={`tab-panel-${key}`}
            id={`tab-${key}`}
            onClick={() => { setTab(key); setFile(false); setFileName(null); }}
            style={{
              flex:1, padding: w < 480 ? "9px 0" : "11px 0", textAlign:"center",
              fontSize: w < 480 ? ".85rem" : ".9rem", fontWeight:500, borderRadius:9,
              background: tab===key ? T.paper2 : "transparent",
              color: tab===key ? T.text : T.dim,
              border:"none", cursor:"pointer",
              boxShadow: tab===key ? "0 4px 12px -4px rgba(0,0,0,.4)" : "none",
              transition:"all .25s ease", fontFamily:"inherit",
            }}>{label}</button>
        ))}
      </div>

      {/* ── SELFIE TAB ── */}
      {tab === "selfie" && (
        <div id="tab-panel-selfie" role="tabpanel" aria-labelledby="tab-selfie">
          <div style={{ position:"relative", width:"100%", aspectRatio:vfAspect, background:"radial-gradient(circle at 50% 40%,#1b1f2c,#0b0d13 75%)", border:`1px solid ${T.frameSoft}`, borderRadius:18, overflow:"hidden", display:"flex", alignItems:"center", justifyContent:"center" }}>
            {/* Bracket corners */}
            <div style={{ position:"relative", width: faceDetected ? "36%" : "42%", paddingBottom: faceDetected ? "45%" : "52%", transition:"all .6s cubic-bezier(.2,.8,.2,1)" }}>
              {[
                { top:0,    left:0,  borderRight:"none", borderBottom:"none" },
                { top:0,    right:0, borderLeft:"none",  borderBottom:"none" },
                { bottom:0, left:0,  borderRight:"none", borderTop:"none"    },
                { bottom:0, right:0, borderLeft:"none",  borderTop:"none"    },
              ].map((s,i) => (
                <span key={i} style={{ position:"absolute", width:20, height:20, border:`2.5px solid ${faceDetected ? T.conf : T.faint}`, transition:"border-color .4s ease", ...s }} />
              ))}
            </div>
            {/* Face silhouette */}
            {faceDetected && (
              <div style={{ position:"absolute", width:"22%", aspectRatio:1, borderRadius:"50%", background:"radial-gradient(circle at 38% 32%,rgba(255,255,255,.14),rgba(255,255,255,.02) 60%),#2a2f3d", top:"32%", left:"50%", transform:"translateX(-50%)" }} />
            )}
            {/* Scan line */}
            {faceDetected && (
              <div style={{ position:"absolute", left:0, right:0, height:2, top:"50%", background:`linear-gradient(90deg,transparent,${T.signal2},${T.conf},${T.signal2},transparent)`, boxShadow:"0 0 12px 2px rgba(139,123,255,.5)", animation:"lscanline 2s ease-in-out infinite" }} />
            )}
            <span aria-live="polite" aria-atomic="true" style={{ position:"absolute", bottom:14, left:"50%", transform:"translateX(-50%)", fontFamily:MONO, fontSize: w < 480 ? ".65rem" : ".74rem", color: faceDetected ? T.conf : T.faint, background:"rgba(0,0,0,.45)", padding:"5px 12px", borderRadius:20, letterSpacing:".03em", whiteSpace:"nowrap", transition:"color .4s" }}>
              {faceDetected ? "Face detected ✓" : "Looking for a face…"}
            </span>
          </div>

          {/* Shutter button */}
          <div style={{ display:"flex", justifyContent:"center", marginTop:22 }}>
            <button onClick={handleShutter} disabled={!faceDetected} aria-label="Capture" style={{
              width: w < 480 ? 54 : 62, height: w < 480 ? 54 : 62, borderRadius:"50%",
              background:T.paper, border:`3px solid ${faceDetected ? T.signal2 : T.frame}`,
              display:"flex", alignItems:"center", justifyContent:"center",
              cursor: faceDetected ? "pointer" : "default",
              opacity: faceDetected ? 1 : 0.45,
              transition:"all .3s ease",
            }}>
              <span style={{ width: w < 480 ? 38 : 46, height: w < 480 ? 38 : 46, borderRadius:"50%", background: faceDetected ? T.signal2 : T.faint, display:"block", transition:"background .3s" }} />
            </button>
          </div>
        </div>
      )}

      {/* ── UPLOAD TAB ── */}
      {tab === "upload" && (
        <div id="tab-panel-upload" role="tabpanel" aria-labelledby="tab-upload">
          <div onClick={() => fileInputRef.current?.click()} style={{
            width:"100%", aspectRatio: w < 480 ? "3/2" : "4/3",
            border:`2px ${fileReady ? "solid" : "dashed"} ${fileReady ? T.conf : T.frame}`,
            borderRadius:18, display:"flex", flexDirection:"column",
            alignItems:"center", justifyContent:"center", gap:10,
            color: fileReady ? T.conf : T.faint,
            cursor:"pointer", transition:"border-color .3s,background .3s",
            background: fileReady ? "rgba(52,211,153,.04)" : "transparent",
            padding:"24px 16px",
          }}
            onMouseEnter={e => { if (!fileReady) e.currentTarget.style.borderColor = T.signal2; }}
            onMouseLeave={e => { if (!fileReady) e.currentTarget.style.borderColor = T.frame; }}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 16V4M12 4l-5 5M12 4l5 5M4 20h16"/></svg>
            <b style={{ ...R, fontSize: w < 480 ? ".9rem" : "1rem", color: fileReady ? T.conf : T.text }}>
              {fileReady ? fileName : "Tap to upload a photo"}
            </b>
            <span style={{ fontSize:".8rem", textAlign:"center" }}>{fileReady ? "Tap to choose a different photo" : "Supports JPG, PNG, WEBP"}</span>
          </div>
          <input ref={fileInputRef} type="file" accept="image/*" style={{ display:"none" }} onChange={handleFileChange} />
          <div style={{ marginTop:18 }}>
            <Btn solid onClick={handleUploadContinue} disabled={!fileReady} aria-label={fileReady ? `Continue with ${fileName}` : "Continue. Upload a photo first"} style={{ width:"100%" }}>Continue →</Btn>
          </div>
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════
   STEP 3 — Processing radar → navigate to selfie page
════════════════════════════════════════════════════ */
function StepProcessing({ event, selfieBlob }) {
  const w = useW();
  const [count, setCount] = useState(0);
  const navigate = useNavigate();
  const total = event?.total_photos || 4;
  const radarSize = w < 480 ? 150 : 190;

  useEffect(() => {
    let frame, val = 0;
    const step = () => {
      val += 0.08;
      setCount(Math.min(Math.round(val), total));
      if (val < total) { frame = requestAnimationFrame(step); }
      else {
        setTimeout(() => {
          navigate(`/attend/${event.id}/selfie`, { state: { event, selfieBlob } });
        }, 600);
      }
    };
    const t = setTimeout(() => { frame = requestAnimationFrame(step); }, 300);
    return () => { clearTimeout(t); cancelAnimationFrame(frame); };
  }, [event, selfieBlob, navigate, total]);

  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", flex:1, padding:`${w < 480 ? 40 : 60}px 24px` }}>
      {/* Radar */}
      <div style={{ position:"relative", width:radarSize, height:radarSize, display:"flex", alignItems:"center", justifyContent:"center", marginBottom:24 }}>
        {[0, Math.round(radarSize*0.136), Math.round(radarSize*0.273)].map(inset => (
          <div key={inset} style={{ position:"absolute", inset, borderRadius:"50%", border:`1px solid ${T.frameSoft}` }} />
        ))}
        <div style={{ position:"absolute", inset:0, borderRadius:"50%", background:"conic-gradient(from 0deg, rgba(109,94,245,0) 0deg, rgba(109,94,245,.55) 58deg, transparent 62deg)", animation:"lradar 1.4s linear infinite" }} />
        <div style={{ position:"relative", width: w < 480 ? 46 : 60, height: w < 480 ? 46 : 60, borderRadius:"50%", background:"linear-gradient(135deg,#6d5ef5,#8b7bff)", zIndex:2, boxShadow:"0 0 30px -4px rgba(109,94,245,.65)" }} />
      </div>
      <p style={{ color:T.dim, fontSize: w < 480 ? ".88rem" : ".95rem", marginBottom:12, textAlign:"center", maxWidth:320 }}>
        Comparing your face against {event?.title || "event photos"}…
      </p>
      <div aria-live="polite" aria-atomic="true" style={{ display:"flex", alignItems:"baseline", gap:8, fontFamily:MONO, fontSize:".9rem", color:T.dim }}>
        <span>Checked</span>
        <b style={{ fontSize: w < 480 ? "1.1rem" : "1.3rem", color:T.text, ...R }}>{count}</b>
        <span>/ {total} photos</span>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════
   ROOT PAGE
════════════════════════════════════════════════════ */
export default function AttendeeEntry() {
  const w = useW();
  const [step, setStep]   = useState(1);
  const [event, setEvent] = useState(null);
  const [selfie, setSelfie] = useState(null);
  const navigate = useNavigate();

  const animateTo = async (n) => {
    const { gsap } = await import("gsap");
    const el = document.getElementById("ls-flow-step");
    if (!el) { setStep(n); return; }
    gsap.to(el, { opacity:0, y:-14, duration:.3, ease:"power2.in", onComplete: () => {
      setStep(n);
      gsap.fromTo(el, { opacity:0, y:14 }, { opacity:1, y:0, duration:.45, ease:"power3.out" });
    }});
  };

  const goStep1ToStep2 = (ev) => { setEvent(ev); animateTo(2); };
  const goStep2ToStep3 = (ev, blob) => { setEvent(ev); setSelfie(blob); animateTo(3); };

  const handleBack = () => { if (step > 1) animateTo(step - 1); else navigate("/"); };

  /* Header padding responsive */
  const hPad = w < 480 ? "18px 16px" : w < 720 ? "22px 24px" : "26px 40px";
  /* main padding */
  const mPad = w < 480 ? "16px 16px 80px" : "20px 24px 90px";

  return (
    <div style={{ minHeight:"100vh", display:"flex", flexDirection:"column", position:"relative", background:T.ink, color:T.text, fontFamily:"'Inter',system-ui,sans-serif", lineHeight:1.6, WebkitFontSmoothing:"antialiased", overflowX:"hidden" }}>
      <style>{`
        @keyframes lspin     { to { transform:rotate(360deg); } }
        @keyframes lradar    { to { transform:rotate(360deg); } }
        @keyframes lscanline { 0%,100%{ top:35% } 50%{ top:65% } }
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
        <button onClick={handleBack} aria-label={step === 1 ? "Back to home" : `Back to step ${step - 1}`}
          style={{ fontSize: w < 480 ? ".82rem" : ".9rem", color:T.dim, background:"none", border:"none", cursor:"pointer", fontFamily:"inherit", transition:"color .2s", flexShrink:0, whiteSpace:"nowrap" }}
          onMouseEnter={e => e.currentTarget.style.color = T.text}
          onMouseLeave={e => e.currentTarget.style.color = T.dim}>
          ← {w < 400 ? "" : step === 1 ? "Home" : "Back"}
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
          <Stepper step={step} />
        </div>
      </header>

      {/* ── Main ── */}
      <main id="ls-flow-main" style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent: step === 3 ? "flex-start" : "center", padding:mPad, position:"relative", zIndex:2 }}>
        <div id="ls-flow-step" style={{ width:"100%", maxWidth: w < 600 ? "100%" : 540, display:"flex", flexDirection:"column", alignItems:"center" }}>
          {step === 1 && <StepCode onNext={goStep1ToStep2} />}
          {step === 2 && <StepVerify event={event} onNext={goStep2ToStep3} />}
          {step === 3 && <StepProcessing event={event} selfieBlob={selfie} />}
        </div>
      </main>
    </div>
  );
}
