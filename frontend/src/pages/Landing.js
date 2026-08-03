import React, { useEffect, useState } from "react";
import { T } from "../components/landing/shared";
import Navbar        from "../components/landing/Navbar";
import Hero          from "../components/landing/Hero";
import Features      from "../components/landing/Features";
import HowItWorks    from "../components/landing/HowItWorks";
import Testimonials  from "../components/landing/Testimonials";
import Pricing       from "../components/landing/Pricing";
import FAQ           from "../components/landing/FAQ";
import CtaPanel      from "../components/landing/CtaPanel";
import LandingFooter from "../components/landing/LandingFooter";

export default function Landing() {
  const [scrolled, setScrolled] = useState(false);

  /* ── Scroll watcher for navbar ── */
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  /* ── GSAP scroll-reveal for all [data-reveal] elements ── */
  useEffect(() => {
    let ctx, cancelled = false;
    const init = async () => {
      const { gsap } = await import("gsap");
      const { ScrollTrigger } = await import("gsap/ScrollTrigger");
      gsap.registerPlugin(ScrollTrigger);
      if (cancelled) return;
      ctx = gsap.context(() => {
        if (window.matchMedia("(prefers-reduced-motion:reduce)").matches) return;
        gsap.utils.toArray("[data-reveal]").forEach(el =>
          gsap.fromTo(el,
            { opacity: 0, y: 24 },
            { opacity: 1, y: 0, duration: .7, ease: "power3.out",
              scrollTrigger: { trigger: el, start: "top 87%" } }
          )
        );
      });
    };
    init();
    return () => { cancelled = true; if (ctx) ctx.revert(); };
  }, []);

  function Div() {
    return <div className="lw"><div style={{ height: 1, background: T.frameSoft }} /></div>;
  }

  return (
    <div style={{ background: T.ink, color: T.text, fontFamily: "'Inter',system-ui,sans-serif", lineHeight: 1.6, WebkitFontSmoothing: "antialiased", minHeight: "100vh", overflowX: "hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&family=Instrument+Sans:wght@500;600;700&display=swap');
        .lw  { max-width:1180px; margin:0 auto; padding:0 32px }
        .lhg { display:grid; grid-template-columns:1.05fr .95fr; gap:64px; align-items:center }
        .lfg { display:grid; grid-template-columns:repeat(3,1fr); gap:20px }
        .lsg { display:grid; grid-template-columns:repeat(3,1fr); gap:0; position:relative }
        .lsg::before { content:""; position:absolute; top:19px; left:8%; right:8%; height:1px; background:rgba(255,255,255,.08) }
        .ltg { display:grid; grid-template-columns:repeat(3,1fr); gap:20px }
        .lpg { display:grid; grid-template-columns:repeat(3,1fr); gap:20px }
        .lft { display:grid; grid-template-columns:1.4fr 1fr 1fr 1fr; gap:40px; margin-bottom:48px }
        .lfc:hover { transform:translateY(-4px); border-color:#262b38 !important }
        .lna a { color:#9a9dae; text-decoration:none; transition:color .2s }
        .lna a:hover { color:#f1efe9 }
        @media(max-width:900px) {
          .lhg,.ltg,.lpg { grid-template-columns:1fr }
          .lfg { grid-template-columns:1fr 1fr }
          .lsg { grid-template-columns:1fr; gap:36px }
          .lsg::before { display:none }
          .lft { grid-template-columns:1fr 1fr }
          .lna { display:none !important }
        }
        @media(max-width:600px) {
          .lfg,.lft { grid-template-columns:1fr }
          .lw { padding:0 18px }
        }
      `}</style>

      <Navbar scrolled={scrolled} />
      <Hero />
      <Div /><Features />
      <Div /><HowItWorks />
      <Div /><Testimonials />
      <Div /><Pricing />
      <Div /><FAQ />
      <CtaPanel />
      <LandingFooter />
    </div>
  );
}
