import React from "react";
import { ScanFace } from "lucide-react";
import { Link } from "react-router-dom";

export default function Brand({ dark = false }) {
  return (
    <Link className={`brand ${dark ? "brand--dark" : ""}`} to="/" aria-label="EventSnap home">
      <span className="brand__mark"><ScanFace size={20} strokeWidth={2.75} /></span>
      <span>Event<span>Snap</span></span>
    </Link>
  );
}
