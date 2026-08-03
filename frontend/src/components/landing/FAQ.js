import React, { useEffect, useRef, useState } from "react";
import { Eyebrow, T, MONO, R } from "./shared";

const FAQS = [
  { q: "Does the AI compare me against faces from other events?",  a: "No. Matching is scoped to a single event index. Your selfie is only ever compared against photos uploaded to the gallery whose code you entered." },
  { q: "What happens to my selfie after matching?",               a: "It is converted into a face embedding — a set of numbers used purely for comparison. The image itself is not kept in storage after the match completes." },
  { q: "Can I browse a gallery without an event code?",           a: "No. The event code is what scopes a search to one gallery. Without it, there is nothing to search against." },
  { q: "What if it misses some of my photos?",                    a: "Organizers can rebuild the face index at any time — no re-uploading required — which usually resolves missed photos." },
  { q: "Do organizers need to know how to code?",                 a: "No. Creating an event, uploading photos, and sharing the code is the entire setup." },
];

function FaqItem({ q, a }) {
  const [open, setOpen] = useState(false);
  const bodyRef  = useRef(null);
  const innerRef = useRef(null);
  useEffect(() => {
    if (!bodyRef.current || !innerRef.current) return;
    bodyRef.current.style.height = open ? innerRef.current.offsetHeight + "px" : "0px";
  }, [open]);
  return (
    <div style={{ borderBottom: `1px solid ${T.frameSoft}` }}>
      <button onClick={() => setOpen(o => !o)} style={{ width: "100%", background: "none", border: "none", color: T.text, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "22px 0", fontSize: "1.02rem", fontFamily: "'Instrument Sans',system-ui,sans-serif", fontWeight: 500, cursor: "pointer", textAlign: "left" }}>
        <span>{q}</span>
        <span style={{ fontFamily: MONO, fontSize: "1.1rem", color: T.signal2, flexShrink: 0, marginLeft: 20, transition: "transform .3s", display: "inline-block", transform: open ? "rotate(45deg)" : "rotate(0)" }}>+</span>
      </button>
      <div ref={bodyRef} style={{ height: 0, overflow: "hidden", transition: "height .35s ease" }}>
        <div ref={innerRef} style={{ paddingBottom: 22, color: T.dim, fontSize: ".95rem", maxWidth: 640 }}>{a}</div>
      </div>
    </div>
  );
}

export default function FAQ() {
  return (
    <section id="faq" style={{ padding: "112px 0" }}>
      <div className="lw">
        <div data-reveal style={{ maxWidth: 640, marginBottom: 56 }}>
          <Eyebrow style={{ marginBottom: 14 }}>Questions</Eyebrow>
          <h2 style={{ ...R, fontSize: "clamp(1.8rem,3vw,2.5rem)", margin: "14px 0" }}>
            Asked before the first upload
          </h2>
        </div>
        <div style={{ maxWidth: 760 }}>
          {FAQS.map((f, i) => <FaqItem key={i} q={f.q} a={f.a} />)}
        </div>
      </div>
    </section>
  );
}
