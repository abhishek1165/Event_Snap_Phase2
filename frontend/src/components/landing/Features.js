import React from "react";
import { Eyebrow, T, DISPLAY, R } from "./shared";

const FEATURES = [
  { icon: "⚡", title: "Matched in under 3 seconds",    desc: "AI matching returns results in under three seconds, whether the gallery has 40 photos or 40,000." },
  { icon: "🔒", title: "We keep the match, not the face", desc: "Only face embeddings are stored. No raw biometric images sit on a server waiting to be a headline." },
  { icon: "◻",  title: "Every event, its own room",      desc: "Searches are isolated per event. A match in one gallery has no way to leak into another." },
  { icon: "📦", title: "Hundreds of photos, one drop",   desc: "Batch upload with background processing, so indexing keeps up while you keep uploading." },
  { icon: "📊", title: "Watch it process, live",         desc: "Real-time indexing status, so you know the exact moment a gallery is ready for attendees." },
  { icon: "📈", title: "See who found what",             desc: "Search volume, match rates, and coverage per event — enough to know if the gallery is working." },
];

export default function Features() {
  return (
    <section id="features" style={{ padding: "112px 0" }}>
      <div className="lw">
        <div data-reveal style={{ maxWidth: 640, marginBottom: 56 }}>
          <Eyebrow style={{ marginBottom: 14 }}>Built for event day</Eyebrow>
          <h2 style={{ ...R, fontSize: "clamp(1.8rem,3vw,2.5rem)", margin: "14px 0" }}>
            Six things that matter more than a demo reel
          </h2>
          <p style={{ color: T.dim, fontSize: "1.05rem" }}>
            Nobody grades this by how it looks in a pitch. They grade it by whether it holds up with 4,000 photos and 300 people trying to download at once.
          </p>
        </div>

        <div className="lfg">
          {FEATURES.map((f, i) => (
            <div key={i} data-reveal className="lfc" style={{
              background: T.paper, border: `1px solid ${T.frameSoft}`,
              borderRadius: 18, padding: 28, transition: "transform .3s,border-color .3s",
            }}>
              <div style={{ width: 38, height: 38, borderRadius: 10, background: T.paper2, border: `1px solid ${T.frameSoft}`, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18, fontSize: "1rem" }}>
                {f.icon}
              </div>
              <h3 style={{ fontFamily: DISPLAY, fontWeight: 600, fontSize: "1.08rem", marginBottom: 10 }}>{f.title}</h3>
              <p style={{ color: T.dim, fontSize: ".95rem" }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
