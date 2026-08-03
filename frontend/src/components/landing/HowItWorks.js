import React from "react";
import { Eyebrow, T, MONO, R } from "./shared";

const STEPS = [
  { n: "01 / 03", title: "Get the event code",  desc: "The organizer hands out one code. Eight characters, scoped to that single event only." },
  { n: "02 / 03", title: "Show your face",       desc: "A selfie, nothing else required. It becomes a match key, not a photo anyone keeps." },
  { n: "03 / 03", title: "Get your frames",      desc: "Every shot you are actually in, ranked by confidence, ready to select and download." },
];

export default function HowItWorks() {
  return (
    <section id="how" style={{ padding: "112px 0" }}>
      <div className="lw">
        <div data-reveal style={{ maxWidth: 640, marginBottom: 56 }}>
          <Eyebrow style={{ marginBottom: 14 }}>The whole flow</Eyebrow>
          <h2 style={{ ...R, fontSize: "clamp(1.8rem,3vw,2.5rem)", margin: "14px 0" }}>
            Three steps, one code, one selfie
          </h2>
          <p style={{ color: T.dim, fontSize: "1.05rem" }}>
            Nothing to install, nothing to name yourself in. This is the entire attendee experience.
          </p>
        </div>

        <div className="lsg">
          {STEPS.map(({ n, title, desc }, i) => (
            <div key={i} data-reveal style={{ paddingRight: 28 }}>
              <span style={{ fontFamily: MONO, fontSize: ".78rem", color: T.signal2, background: T.ink, display: "inline-block", paddingRight: 10, position: "relative", zIndex: 1, marginBottom: 22 }}>
                {n}
              </span>
              <h3 style={{ ...R, fontSize: "1.15rem", marginBottom: 10 }}>{title}</h3>
              <p style={{ color: T.dim, fontSize: ".95rem" }}>{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
