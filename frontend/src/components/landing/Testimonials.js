import React from "react";
import { Eyebrow, T, MONO, R } from "./shared";

const CARDS = [
  { q: '"We handed out one eight-character code at check-in and never touched a spreadsheet of names again."', name: "Priya M.",  role: "Wedding Photographer" },
  { q: '"Around 300 people found their own photos before the after-party was even over."',                     name: "Arjun K.", role: "Conference Organizer" },
  { q: '"The confidence score is what sold our team. No more guessing which blurry face is which."',          name: "Sneha R.", role: "College Fest Lead" },
];

export default function Testimonials() {
  return (
    <section style={{ padding: "112px 0" }}>
      <div className="lw">
        <div data-reveal style={{ maxWidth: 640, marginBottom: 56 }}>
          <Eyebrow style={{ marginBottom: 14 }}>What organizers said</Eyebrow>
          <h2 style={{ ...R, fontSize: "clamp(1.8rem,3vw,2.5rem)", margin: "14px 0" }}>
            Not case studies. Just what happened after one code went out.
          </h2>
        </div>
        <div className="ltg">
          {CARDS.map(({ q, name, role }, i) => (
            <div key={i} data-reveal style={{ background: T.paper, border: `1px solid ${T.frameSoft}`, borderRadius: 18, padding: 26, display: "flex", flexDirection: "column", gap: 18 }}>
              <p style={{ color: T.text, fontSize: ".98rem", lineHeight: 1.65 }}>{q}</p>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: "auto" }}>
                <div style={{ width: 34, height: 34, borderRadius: "50%", background: "linear-gradient(135deg,#6d5ef5,#8b7bff)", flexShrink: 0 }} />
                <div>
                  <b style={{ display: "block", fontSize: ".88rem" }}>{name}</b>
                  <span style={{ fontFamily: MONO, fontSize: ".72rem", color: T.faint }}>{role}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
