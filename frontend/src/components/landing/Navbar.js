import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { LogoMark, T, R, DISPLAY } from "./shared";

/**
 * Navbar — fixed top nav for the landing page.
 * Uses IntersectionObserver on a sentinel to detect "scrolled" (NOT
 * window.addEventListener('scroll') — banned per design-taste §5.D).
 * Glass background only when scrolled (blur signals dismissal).
 */
export default function Navbar() {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);
  const sentinel = useRef(null);

  useEffect(() => {
    const el = sentinel.current;
    if (!el) return;
    // fires once the sentinel (at very top) leaves the viewport
    const io = new IntersectionObserver(
      ([entry]) => setScrolled(!entry.isIntersecting),
      { threshold: 0 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const links = [
    ["#how", "How it works"],
    ["#features", "Features"],
    ["#pricing", "Pricing"],
    ["#faq", "FAQ"],
  ];

  return (
    <>
      <div ref={sentinel} aria-hidden="true" style={{ height: 1 }} />
      <header
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 50,
          height: 68,
          borderBottom: `1px solid ${scrolled ? T.frameSoft : "transparent"}`,
          background: scrolled ? "rgba(8,9,13,.72)" : "transparent",
          backdropFilter: scrolled ? "blur(16px)" : "none",
          WebkitBackdropFilter: scrolled ? "blur(16px)" : "none",
          transition: "background .3s ease, border-color .3s ease, backdrop-filter .3s ease",
        }}
      >
        <nav
          className="lw"
          aria-label="Main navigation"
          style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: "100%" }}
        >
          {/* Brand */}
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            aria-label="Event Snap home, scroll to top"
            style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", background: "none", border: "none", color: "inherit", padding: 0, ...R, fontFamily: DISPLAY, fontSize: "1.02rem" }}
          >
            <LogoMark size={24} />
            Event&nbsp;Snap
          </button>

          {/* Anchor links — one line at desktop, hidden < 900px */}
          <div className="lna" style={{ display: "flex", gap: 32, fontSize: ".9rem", color: T.dim }}>
            {links.map(([h, l]) => (
              <a key={h} href={h} style={{ textDecoration: "none", color: T.dim, transition: "color .2s ease" }}
                onMouseEnter={(e) => (e.currentTarget.style.color = T.text)}
                onMouseLeave={(e) => (e.currentTarget.style.color = T.dim)}
              >
                {l}
              </a>
            ))}
          </div>

          {/* Actions — single intent each (no duplicate CTA) */}
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <a
              href="/auth"
              onClick={(e) => { e.preventDefault(); navigate("/auth"); }}
              style={{ fontSize: ".9rem", color: T.dim, textDecoration: "none", transition: "color .2s ease" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = T.text)}
              onMouseLeave={(e) => (e.currentTarget.style.color = T.dim)}
            >
              Sign in
            </a>
            <button
              onClick={() => navigate("/auth")}
              style={{
                display: "inline-flex", alignItems: "center", gap: 8,
                padding: "9px 16px", borderRadius: 9, border: "none",
                background: T.signal, color: "#fff", fontWeight: 600, fontSize: ".85rem",
                cursor: "pointer", fontFamily: "inherit",
                boxShadow: "0 1px 0 0 rgba(255,255,255,.16) inset, 0 6px 18px -10px rgba(109,94,245,.45)",
                transition: "transform .2s cubic-bezier(.16,1,.3,1), background .2s ease",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.transform = "translateY(-1px)"; e.currentTarget.style.background = T.signal2; }}
              onMouseLeave={(e) => { e.currentTarget.style.transform = "none"; e.currentTarget.style.background = T.signal; }}
            >
              Get started
            </button>
          </div>
        </nav>
      </header>
    </>
  );
}
