import React, { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Eyebrow, Btn, LogoMark, T, MONO, DISPLAY, R } from "./shared";
import Viewfinder from "./Viewfinder";

export default function Hero({ scrolled }) {
  const navigate   = useNavigate();
  const glowRef    = useRef(null);
  const sectionRef = useRef(null);

  // ── Mouse-follow glow animation ──────────────────────────────────────
  useEffect(() => {
    let gsap, xTo, yTo, cancelled = false;
    const init = async () => {
      const mod = await import("gsap");
      gsap = mod.gsap || mod.default;
      if (cancelled || !glowRef.current || !sectionRef.current) return;
      if (window.matchMedia("(prefers-reduced-motion:reduce)").matches) return;

      xTo = gsap.quickTo(glowRef.current, "x", { duration: 1.4, ease: "power3" });
      yTo = gsap.quickTo(glowRef.current, "y", { duration: 1.4, ease: "power3" });

      const onMove = (e) => {
        const r = sectionRef.current.getBoundingClientRect();
        xTo(e.clientX - r.left - 320);
        yTo(e.clientY - r.top  - 320);
      };
      sectionRef.current.addEventListener("mousemove", onMove);
      return () => { if (sectionRef.current) sectionRef.current.removeEventListener("mousemove", onMove); };
    };
    let cleanup;
    init().then(fn => { cleanup = fn; });
    return () => { cancelled = true; if (cleanup) cleanup(); };
  }, []);

  // ── Hero entrance + stat count-up ────────────────────────────────────
  useEffect(() => {
    let ctx, cancelled = false;
    const init = async () => {
      const mod = await import("gsap");
      const gsap = mod.gsap || mod.default;
      if (cancelled) return;
      ctx = gsap.context(() => {
        if (window.matchMedia("(prefers-reduced-motion:reduce)").matches) return;
        gsap.timeline({ defaults: { ease: "power3.out" } })
          .from("[data-hero-eye]",  { opacity: 0, y: 10, duration: .5 })
          .from("[data-hero-h1]",   { opacity: 0, y: 22, duration: .7 }, "-=.3")
          .from("[data-hero-lead]", { opacity: 0, y: 18, duration: .6 }, "-=.4")
          .from("[data-hero-btn]",  { opacity: 0, y: 14, stagger: .1, duration: .5 }, "-=.35")
          .from("[data-hero-stat]", { opacity: 0, y: 14, stagger: .1, duration: .5 }, "-=.3")
          .from("[data-hero-vf]",   { opacity: 0, y: 24, duration: .8 }, "-=.9");

        const proxy = { photos: 0, speed: 0, conf: 0 };
        gsap.to(proxy, { photos: 50, speed: 3, conf: 98.7, duration: 1.8, delay: .6, ease: "power2.out",
          onUpdate: () => {
            const p = document.getElementById("ls-photos");
            const s = document.getElementById("ls-speed");
            const c = document.getElementById("ls-conf");
            if (p) p.textContent = Math.round(proxy.photos);
            if (s) s.textContent = Math.round(proxy.speed);
            if (c) c.textContent  = proxy.conf.toFixed(1);
          },
        });
      });
    };
    init();
    return () => { cancelled = true; if (ctx) ctx.revert(); };
  }, []);

  return (
    <section ref={sectionRef} style={{ position: "relative", padding: "168px 0 120px", overflow: "hidden" }}>
      {/* ── Mouse-follow radial glow ── */}
      <div ref={glowRef} style={{
        position: "absolute", width: 640, height: 640, borderRadius: "50%",
        background: "radial-gradient(circle,rgba(109,94,245,.22),transparent 65%)",
        top: -260, left: -180, pointerEvents: "none", filter: "blur(10px)",
        willChange: "transform",
      }} />

      {/* ── Ambient secondary glow (static) ── */}
      <div style={{
        position: "absolute", width: 400, height: 400, borderRadius: "50%",
        background: "radial-gradient(circle,rgba(52,211,153,.08),transparent 65%)",
        bottom: -100, right: -50, pointerEvents: "none", filter: "blur(24px)",
      }} />

      <div className="lw">
        <div className="lhg">
          {/* LEFT */}
          <div>
            <div data-hero-eye>
              <Eyebrow style={{ marginBottom: 22 }}>Face matching for event photography</Eyebrow>
            </div>
            <h1 data-hero-h1 style={{ ...R, fontSize: "clamp(2.5rem,4.6vw,3.9rem)", lineHeight: 1.06, margin: "22px 0" }}>
              Your face is<br />the{" "}
              <span style={{ background: "linear-gradient(100deg,#8b7bff,#34d399 90%)", WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>
                search bar.
              </span>
            </h1>
            <p data-hero-lead style={{ fontSize: "1.14rem", color: T.dim, maxWidth: 480, marginBottom: 34 }}>
              One selfie in. Every photo you are actually in, out — matched by confidence, not by luck, in the time it takes to read this sentence.
            </p>
            <div data-hero-btn style={{ display: "flex", gap: 14, marginBottom: 52, flexWrap: "wrap" }}>
              <Btn solid onClick={() => navigate("/auth")}>For organizers →</Btn>
              <Btn onClick={() => navigate("/attendjoin")}>Find my photos</Btn>
            </div>
            <div style={{ display: "flex", gap: 40, flexWrap: "wrap" }}>
              {[
                { id: "ls-photos", pre: "",  suf: "K+", label: "Photos scanned", accessible: "50K+ Photos scanned" },
                { id: "ls-speed",  pre: "<", suf: "s",  label: "Median match time", accessible: "Under 3 seconds Median match time" },
                { id: "ls-conf",   pre: "",  suf: "%",  label: "Typical confidence", accessible: "98.7% Typical confidence" },
              ].map(({ id, pre, suf, label, accessible }) => (
                <div key={id} data-hero-stat aria-label={accessible}>
                  <b style={{ display: "block", ...R, fontSize: "1.7rem" }} aria-hidden="true">
                    {pre}<span id={id}>0</span>{suf}
                  </b>
                  <span style={{ fontFamily: MONO, fontSize: ".72rem", letterSpacing: ".06em", color: T.faint, textTransform: "uppercase" }}>{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* RIGHT — Viewfinder */}
          <div data-hero-vf><Viewfinder /></div>
        </div>
      </div>
    </section>
  );
}
