import React from "react";
import { LogoMark, T, MONO, R } from "./shared";

const NAV = [
  ["Product",  [["How it works","#how"],["Features","#features"],["Pricing","#pricing"]]],
  ["Company",  [["About","#"],["Contact","#"]]],
  ["Legal",    [["Privacy","#"],["Terms","#"]]],
];

export default function LandingFooter() {
  return (
    <footer style={{ padding: "64px 0 40px", borderTop: `1px solid ${T.frameSoft}` }}>
      <div className="lw">
        <div className="lft">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, ...R, fontSize: "1.05rem" }}>
              <LogoMark />FaceShot
            </div>
            <p style={{ color: T.faint, fontSize: ".88rem", marginTop: 14, maxWidth: 260 }}>
              Face matching for event photography. One code in, every photo of you out.
            </p>
          </div>
          {NAV.map(([head, links]) => (
            <div key={head}>
              <h4 style={{ fontFamily: MONO, fontSize: ".72rem", letterSpacing: ".08em", textTransform: "uppercase", color: T.faint, marginBottom: 16 }}>{head}</h4>
              {links.map(([label, href]) => (
                <a key={label} href={href} style={{ display: "block", color: T.dim, fontSize: ".9rem", marginBottom: 12, textDecoration: "none", transition: "color .2s" }}
                  onMouseEnter={e => e.currentTarget.style.color = T.text}
                  onMouseLeave={e => e.currentTarget.style.color = T.dim}>
                  {label}
                </a>
              ))}
            </div>
          ))}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", color: T.faint, fontSize: ".82rem", fontFamily: MONO, flexWrap: "wrap", gap: 10 }}>
          <span>© 2026 FaceShot</span>
          <span>Built for the day of the event</span>
        </div>
      </div>
    </footer>
  );
}
