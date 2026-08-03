import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Eyebrow, Btn, T, MONO, R } from "./shared";

const PLANS = [
  { title: "Starter",    desc: "For one event at a time.",                  price: { m: "Free",   y: "Free"   }, features: ["1 active event","Up to 200 photos","Standard match speed","Community support"],                                            solid: false, cta: "Start free" },
  { title: "Studio",     desc: "For recurring events and full seasons.",     price: { m: "$24",    y: "$19"    }, features: ["Unlimited events","Up to 5,000 photos / event","Priority match speed (<3s)","Analytics dashboard","Email support"],        solid: true,  cta: "Start free trial", popular: true },
  { title: "Enterprise", desc: "For venues, agencies, and platforms.",       price: { m: "Custom", y: "Custom" }, features: ["Unlimited photos","Dedicated infrastructure","SSO & audit logs","Uptime SLA","Dedicated account manager"],                  solid: false, cta: "Talk to us" },
];

export default function Pricing() {
  const navigate  = useNavigate();
  const [yearly, setYearly] = useState(false);

  return (
    <section id="pricing" style={{ padding: "112px 0" }}>
      <div className="lw">
        <div data-reveal style={{ maxWidth: 640, marginBottom: 56 }}>
          <Eyebrow style={{ marginBottom: 14 }}>Pricing</Eyebrow>
          <h2 style={{ ...R, fontSize: "clamp(1.8rem,3vw,2.5rem)", margin: "14px 0" }}>
            Scales with your events, not your headcount
          </h2>
          <p style={{ color: T.dim, fontSize: "1.05rem" }}>
            No per-seat pricing. No per-attendee pricing. Just how many events you are actually running.
          </p>
        </div>

        {/* Toggle */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 16, marginBottom: 52, fontSize: ".92rem", color: T.dim }}>
          <span>Monthly</span>
          <button onClick={() => setYearly(y => !y)} style={{ width: 46, height: 26, borderRadius: 20, background: yearly ? T.signal : T.paper2, border: `1px solid ${yearly ? T.signal : T.frame}`, position: "relative", cursor: "pointer", padding: 0, transition: "background .3s,border-color .3s" }}>
            <div style={{ position: "absolute", top: 2, left: 2, width: 20, height: 20, borderRadius: "50%", background: T.text, transition: "transform .3s", transform: yearly ? "translateX(20px)" : "none" }} />
          </button>
          <span>Yearly <span style={{ fontFamily: MONO, fontSize: ".66rem", color: T.conf, background: T.confSoft, padding: "2px 8px", borderRadius: 20 }}>Save 20%</span></span>
        </div>

        <div className="lpg">
          {PLANS.map(({ title, desc, price, features, solid, cta, popular }) => (
            <div key={title} style={{ background: popular ? T.paper2 : T.paper, border: `1px solid ${popular ? T.signal2 : T.frameSoft}`, borderRadius: 18, padding: 30, display: "flex", flexDirection: "column", position: "relative" }}>
              {popular && <span style={{ position: "absolute", top: -13, left: 30, background: "linear-gradient(135deg,#6d5ef5,#8b7bff)", fontFamily: MONO, fontSize: ".66rem", letterSpacing: ".06em", textTransform: "uppercase", padding: "5px 12px", borderRadius: 20, color: "#fff" }}>Most organizers</span>}
              <h3 style={{ ...R, fontSize: "1.1rem", marginBottom: 6 }}>{title}</h3>
              <p style={{ color: T.dim, fontSize: ".88rem", marginBottom: 22 }}>{desc}</p>
              <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginBottom: 22 }}>
                <b style={{ ...R, fontSize: "2.4rem" }}>{yearly ? price.y : price.m}</b>
                {price.m !== "Free" && price.m !== "Custom" && <span style={{ color: T.faint, fontSize: ".85rem" }}>/ mo</span>}
              </div>
              <ul style={{ listStyle: "none", padding: 0, display: "flex", flexDirection: "column", gap: 12, marginBottom: 26, flex: 1 }}>
                {features.map((f, i) => (
                  <li key={i} style={{ display: "flex", gap: 10, fontSize: ".88rem", color: T.dim }}>
                    <span style={{ width: 16, height: 16, borderRadius: "50%", background: T.confSoft, flexShrink: 0, marginTop: 2 }} />{f}
                  </li>
                ))}
              </ul>
              <Btn solid={solid} onClick={() => navigate("/auth")} style={{ justifyContent: "center" }}>{cta}</Btn>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
