import React from "react";
import { useNavigate } from "react-router-dom";
import { LogoMark, Btn, T, R } from "./shared";

export default function Navbar({ scrolled }) {
  const navigate = useNavigate();
  return (
    <header style={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 50, borderBottom: `1px solid ${scrolled ? T.frameSoft : "transparent"}`, background: scrolled ? "rgba(8,9,13,.75)" : "transparent", backdropFilter: scrolled ? "blur(14px)" : "none", transition: "background .3s,border-color .3s" }}>
      <nav className="lw" aria-label="Main navigation" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: 76 }}>
        <button onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })} aria-label="FaceShot home, scroll to top" style={{ display: "flex", alignItems: "center", gap: 10, ...R, fontSize: "1.05rem", cursor: "pointer", background: "none", border: "none", color: "inherit", padding: 0 }}>
          <LogoMark />FaceShot
        </button>
        <div className="lna" style={{ display: "flex", gap: 34, fontSize: ".92rem", color: T.dim }}>
          {[["#how","How it works"],["#features","Features"],["#pricing","Pricing"],["#faq","FAQ"]].map(([h, l]) => (
            <a key={h} href={h} style={{ textDecoration: "none", color: T.dim, transition: "color .2s" }}
              onMouseEnter={e => e.currentTarget.style.color = T.text}
              onMouseLeave={e => e.currentTarget.style.color = T.dim}>{l}</a>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          <a href="/auth" onClick={e => { e.preventDefault(); navigate("/auth"); }}
            style={{ fontSize: ".92rem", color: T.dim, textDecoration: "none", transition: "color .2s" }}
            onMouseEnter={e => e.currentTarget.style.color = T.text}
            onMouseLeave={e => e.currentTarget.style.color = T.dim}>
            Sign in
          </a>
          <Btn solid onClick={() => navigate("/auth")} style={{ padding: "9px 16px", fontSize: ".85rem", borderRadius: 9 }}>
            Get started
          </Btn>
        </div>
      </nav>
    </header>
  );
}
