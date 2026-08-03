import React from "react";
import { useNavigate } from "react-router-dom";
import { Btn, T, R } from "./shared";

export default function CtaPanel() {
  const navigate = useNavigate();
  return (
    <section style={{ padding: "0 0 112px" }}>
      <div className="lw">
        <div data-reveal style={{ background: "linear-gradient(135deg,#141227,#0d1a1f)", border: `1px solid ${T.frameSoft}`, borderRadius: 28, padding: "clamp(36px,6vw,72px) clamp(24px,5vw,56px)", textAlign: "center", position: "relative", overflow: "hidden" }}>
          <div style={{ position: "absolute", width: 420, height: 420, background: "radial-gradient(circle,rgba(109,94,245,.25),transparent 65%)", top: -160, right: -100 }} />
          <h2 style={{ ...R, fontSize: "clamp(1.8rem,3.4vw,2.6rem)", marginBottom: 16, position: "relative" }}>
            Send out one code.<br />Stop answering "can you send me that photo?"
          </h2>
          <p style={{ color: T.dim, marginBottom: 34, position: "relative" }}>Set up your first event gallery in under five minutes.</p>
          <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap", position: "relative" }}>
            <Btn solid onClick={() => navigate("/auth")}>Create your first event →</Btn>
            <Btn onClick={() => navigate("/attendjoin")}>Find my photos</Btn>
          </div>
        </div>
      </div>
    </section>
  );
}
