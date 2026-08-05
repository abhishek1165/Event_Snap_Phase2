import React, { useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { Eyebrow, Btn, T, MONO, R } from "./shared";
import Viewfinder from "./Viewfinder";
import { useCountUp } from "@/hooks/useCountUp";

gsap.registerPlugin(useGSAP);

/**
 * Hero — the landing thesis.
 * Design discipline (design-taste §4.7):
 *  - max 4 text elements: eyebrow, headline, subtext, CTAs
 *  - stats strip moved OUT to a dedicated band below (StatBand)
 *  - headline ≤ 2 lines, subtext ≤ 20 words
 *  - top padding ≤ pt-24 desktop
 *
 * Signature: the Viewfinder (aperture scan) on the right = the one
 * memorable element. Everything else stays disciplined.
 */
export default function Hero() {
  const navigate = useNavigate();
  const scope = useRef(null);

  useGSAP(
    () => {
      if (window.matchMedia("(prefers-reduced-motion:reduce)").matches) return;
      const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
      tl.from("[data-hero-eye]", { opacity: 0, y: 10, duration: 0.5 })
        .from("[data-hero-h1]", { opacity: 0, y: 24, duration: 0.7 }, "-=0.3")
        .from("[data-hero-lead]", { opacity: 0, y: 18, duration: 0.6 }, "-=0.4")
        .from("[data-hero-btn]", { opacity: 0, y: 14, stagger: 0.1, duration: 0.5 }, "-=0.35")
        .from("[data-hero-vf]", { opacity: 0, y: 24, duration: 0.8 }, "-=0.9");
    },
    { scope }
  );

  // mouse-follow glow — gsap.quickTo for perf (gsap-performance skill)
  useGSAP(
    () => {
      if (window.matchMedia("(prefers-reduced-motion:reduce)").matches) return;
      const glow = scope.current?.querySelector("[data-hero-glow]");
      if (!glow || !scope.current) return;
      const xTo = gsap.quickTo(glow, "x", { duration: 1.4, ease: "power3" });
      const yTo = gsap.quickTo(glow, "y", { duration: 1.4, ease: "power3" });
      const onMove = (e) => {
        const r = scope.current.getBoundingClientRect();
        xTo(e.clientX - r.left - 320);
        yTo(e.clientY - r.top - 320);
      };
      scope.current.addEventListener("mousemove", onMove);
      return () => scope.current?.removeEventListener("mousemove", onMove);
    },
    { scope }
  );

  return (
    <section ref={scope} style={{ position: "relative", padding: "132px 0 96px", overflow: "hidden" }}>
      {/* mouse-follow radial glow (violet) */}
      <div data-hero-glow style={{
        position: "absolute", width: 640, height: 640, borderRadius: "50%",
        background: "radial-gradient(circle,rgba(109,94,245,.18),transparent 65%)",
        top: -260, left: -180, pointerEvents: "none", filter: "blur(10px)", willChange: "transform",
      }} />
      {/* static secondary mint glow */}
      <div aria-hidden="true" style={{
        position: "absolute", width: 420, height: 420, borderRadius: "50%",
        background: "radial-gradient(circle,rgba(52,211,153,.07),transparent 65%)",
        bottom: -120, right: -60, pointerEvents: "none", filter: "blur(24px)",
      }} />

      <div className="lw">
        <div className="lhg">
          {/* LEFT — message */}
          <div>
            <div data-hero-eye>
              <Eyebrow style={{ marginBottom: 22 }}>Face matching for event photography</Eyebrow>
            </div>
            <h1 data-hero-h1 style={{ ...R, fontSize: "clamp(2.5rem,4.8vw,4rem)", lineHeight: 1.05, margin: "22px 0" }}>
              Your face is<br />
              the{" "}
              <span style={{ background: "linear-gradient(100deg,#8b7bff,#34d399 90%)", WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>
                search bar.
              </span>
            </h1>
            <p data-hero-lead style={{ fontSize: "1.12rem", color: T.dim, maxWidth: 480, marginBottom: 32, lineHeight: 1.6 }}>
              Take one selfie. Get every photo you appear in, matched by face recognition in seconds.
            </p>
            <div data-hero-btn style={{ display: "flex", gap: 14, marginBottom: 0, flexWrap: "wrap" }}>
              <Btn solid onClick={() => navigate("/auth")}>For organizers</Btn>
              <Btn onClick={() => navigate("/attendjoin")}>Find my photos</Btn>
            </div>
          </div>

          {/* RIGHT — signature Viewfinder */}
          <div data-hero-vf><Viewfinder /></div>
        </div>
      </div>
    </section>
  );
}

/* ── StatBand — the stats that used to live inside the hero.
   Moved here per design-taste §4.7 (hero stack discipline: max 4 elements,
   social-proof/metrics belong below the hero, not inside it). */
function StatItem({ value, decimals, prefix, suffix, label }) {
  const ref = useCountUp(value, { decimals, prefix, suffix });
  return (
    <div>
      <b style={{ display: "block", ...R, fontSize: "1.7rem", color: T.text }}>
        <span ref={ref}>0</span>
      </b>
      <span style={{ fontFamily: MONO, fontSize: ".7rem", letterSpacing: ".08em", color: T.faint, textTransform: "uppercase" }}>
        {label}
      </span>
    </div>
  );
}

export function StatBand() {
  return (
    <section style={{ padding: "0 0 72px" }}>
      <div className="lw">
        <div data-reveal style={{
          display: "flex", gap: 48, flexWrap: "wrap",
          borderTop: `1px solid ${T.frameSoft}`, borderBottom: `1px solid ${T.frameSoft}`,
          padding: "28px 0",
        }}>
          <StatItem value={50} prefix="" suffix="K+" label="Photos scanned" />
          <StatItem value={3} prefix="<" suffix="s" label="Median match time" />
          <StatItem value={98.7} decimals={1} prefix="" suffix="%" label="Typical confidence" />
        </div>
      </div>
    </section>
  );
}
