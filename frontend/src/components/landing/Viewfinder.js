import React, { useEffect, useRef, useState } from "react";
import { Eyebrow, T, MONO, DISPLAY } from "./shared";
import { SHADOW } from "@/design/tokens";

const TARGETS = [
  { idx: 1, conf: 98.7 },
  { idx: 6, conf: 96.4 },
  { idx: 10, conf: 94.1 },
];
const TOTAL = 16;

/**
 * Viewfinder — the hero's signature element.
 * A looping aperture scan over a 4x4 grid of faces: a scan-line sweeps,
 * matches lock in with confidence percentages, a live readout counts.
 * The green "Live" dot here conveys real semantic state (scan active),
 * so it passes the §9.F decorative-dot ban.
 */
export default function Viewfinder() {
  const scannerRef = useRef(null);
  const gridRef = useRef(null);
  const counters = useRef({ matched: 0, conf: 0 });
  const [display, setDisplay] = useState({ scanned: 0, matched: 0, conf: "0.0%" });
  const [matched, setMatched] = useState(new Set());

  useEffect(() => {
    let cancelled = false, timer, tl;
    const boot = async () => {
      const { gsap } = await import("gsap");
      if (cancelled) return;
      if (window.matchMedia("(prefers-reduced-motion:reduce)").matches) {
        setMatched(new Set(TARGETS.map((t) => t.idx)));
        setDisplay({ scanned: TOTAL, matched: TARGETS.length, conf: "98.7%" });
        return;
      }
      function loop() {
        if (cancelled) return;
        setMatched(new Set());
        counters.current = { matched: 0, conf: 0 };
        setDisplay({ scanned: 0, matched: 0, conf: "0.0%" });
        const scanner = scannerRef.current;
        if (!scanner) return;
        const stageH = gridRef.current ? gridRef.current.offsetHeight - 2 : 288;
        gsap.set(scanner, { y: 0, opacity: 0 });
        const proxy = { scanned: 0, conf: 0 };
        tl = gsap.timeline({ onComplete: () => { timer = setTimeout(loop, 1600); } });
        tl.to(scanner, { opacity: 1, duration: 0.25 })
          .to(proxy, { scanned: TOTAL, duration: 2.3, ease: "none",
            onUpdate: () => setDisplay((d) => ({ ...d, scanned: Math.round(proxy.scanned) })) }, "<")
          .to(scanner, { y: stageH, duration: 2.3, ease: "none" }, "<");
        TARGETS.forEach(({ idx, conf }, i) => {
          tl.call(() => {
            counters.current.matched += 1;
            const m = counters.current.matched;
            setMatched((prev) => new Set([...prev, idx]));
            gsap.to(proxy, { conf, duration: 0.6, ease: "power2.out",
              onUpdate: () => setDisplay((d) => ({ ...d, matched: m, conf: proxy.conf.toFixed(1) + "%" })) });
          }, null, `<${0.55 + i * 1.15}`);
        });
        tl.to(scanner, { opacity: 0, duration: 0.4 }, "+=0.7");
      }
      loop();
    };
    boot();
    return () => { cancelled = true; clearTimeout(timer); if (tl) tl.kill(); };
  }, []);

  const targetMap = Object.fromEntries(TARGETS.map((t) => [t.idx, t.conf.toFixed(1) + "%"]));

  return (
    <div style={{
      background: T.paper,
      border: `1px solid ${T.frameSoft}`,
      borderRadius: 18,
      padding: 22,
      boxShadow: SHADOW.card,
    }}>
      {/* header — Live status (semantic dot OK) */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <Eyebrow>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            {/* semantic state dot: scan is live */}
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: T.conf, boxShadow: `0 0 0 3px ${T.confSoft}`, flexShrink: 0 }} />
            Live match
          </span>
        </Eyebrow>
        <span style={{ fontFamily: MONO, fontSize: ".7rem", color: T.faint, letterSpacing: ".05em" }}>{TOTAL} FRAMES</span>
      </div>

      {/* grid */}
      <div style={{ background: T.ink, border: `1px solid ${T.frameSoft}`, borderRadius: 10, padding: 14 }}>
        <div ref={gridRef} style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8, height: 288, position: "relative" }}>
          {Array.from({ length: TOTAL }, (_, i) => {
            const isMatch = matched.has(i);
            return (
              <div key={i} style={{
                position: "relative", borderRadius: 8, overflow: "hidden",
                background: "linear-gradient(160deg,#171a24,#0f1119)",
                border: `1px solid ${isMatch ? T.conf : T.frameSoft}`,
                boxShadow: isMatch ? `0 0 0 1px ${T.conf},0 0 18px -2px ${T.confSoft}` : "none",
                transition: "border-color .3s ease, box-shadow .3s ease",
              }}>
                <div style={{
                  position: "absolute", left: "50%", top: "38%", width: "46%", aspectRatio: 1,
                  borderRadius: "50%", transform: "translate(-50%,-50%)",
                  background: "radial-gradient(circle at 35% 30%,rgba(255,255,255,.16),rgba(255,255,255,.02) 60%),#2a2f3d",
                }} />
                {isMatch && (
                  <span style={{
                    position: "absolute", bottom: 4, left: 4, right: 4,
                    fontFamily: MONO, fontSize: ".55rem", textAlign: "center", color: T.conf,
                    background: "rgba(6,10,8,.7)", borderRadius: 4, padding: "2px 0",
                  }}>
                    {targetMap[i]}
                  </span>
                )}
              </div>
            );
          })}
          <div ref={scannerRef} style={{
            position: "absolute", left: 0, right: 0, top: 0, height: 2,
            background: `linear-gradient(90deg,transparent,${T.signal2},${T.conf},${T.signal2},transparent)`,
            boxShadow: "0 0 16px 2px rgba(139,123,255,.55)",
          }} />
        </div>
      </div>

      {/* readouts */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginTop: 14 }}>
        {[["Scanned", display.scanned, false], ["Matched", display.matched, false], ["Confidence", display.conf, true]].map(([label, val, accent]) => (
          <div key={label} style={{ background: T.paper2, border: `1px solid ${T.frameSoft}`, borderRadius: 10, padding: "10px 12px" }}>
            <span style={{ display: "block", fontFamily: MONO, fontSize: ".6rem", letterSpacing: ".08em", color: T.faint, textTransform: "uppercase", marginBottom: 4 }}>{label}</span>
            <b style={{ fontFamily: DISPLAY, fontSize: "1.15rem", color: accent ? T.conf : T.text }}>{val}</b>
          </div>
        ))}
      </div>
    </div>
  );
}
