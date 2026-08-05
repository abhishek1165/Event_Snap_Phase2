import React from "react";
import { useNavigate } from "react-router-dom";
import { Btn, T, R } from "./shared";
import { SHADOW } from "@/design/tokens";

export default function CtaPanel() {
  const navigate = useNavigate();
  return (
    <section style={{ padding: "0 0 112px" }}>
      <div className="lw">
        <div data-reveal style={{
          background: "linear-gradient(135deg,#141227,#0d1a1f)",
          border: `1px solid ${T.frameSoft}`,
          borderRadius: 28,
          padding: "clamp(36px,6vw,72px) clamp(24px,5vw,56px)",
          textAlign: "center",
          position: "relative",
          overflow: "hidden",
          boxShadow: SHADOW.lift,
        }}>
          {/* diffused ambient glow (violet + mint) */}
          <div aria-hidden="true" style={{ position: "absolute", width: 420, height: 420, background: "radial-gradient(circle,rgba(109,94,245,.22),transparent 65%)", top: -160, right: -100, pointerEvents: "none" }} />
          <div aria-hidden="true" style={{ position: "absolute", width: 360, height: 360, background: "radial-gradient(circle,rgba(52,211,153,.12),transparent 65%)", bottom: -140, left: -80, pointerEvents: "none" }} />

          <h2 style={{ ...R, fontSize: "clamp(1.8rem,3.4vw,2.6rem)", marginBottom: 16, position: "relative", lineHeight: 1.1 }}>
            Send one code.<br />Stop chasing "send me that photo".
          </h2>
          <p style={{ color: T.dim, marginBottom: 34, position: "relative", maxWidth: 460, marginLeft: "auto", marginRight: "auto" }}>
            Set up your first event gallery in under five minutes.
          </p>
          <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap", position: "relative" }}>
            <Btn solid onClick={() => navigate("/auth")}>Create your first event</Btn>
            <Btn onClick={() => navigate("/attendjoin")}>Find my photos</Btn>
          </div>
        </div>
      </div>
    </section>
  );
}
