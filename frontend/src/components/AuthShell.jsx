import React from "react";
import { Link } from "react-router-dom";
import { Aperture, Star, Zap } from "lucide-react";
import Brand from "./Brand";

export default function AuthShell({ mode, children }) {
  const signup = mode === "signup";
  return (
    <main className={`auth-shell ${signup ? "auth-shell--signup" : "auth-shell--login"}`}>
      <section className="auth-shell__form">
        <Brand />
        <div className="auth-shell__content">{children}</div>
        <p className="auth-shell__note">Facial recognition and instant matching powered by Event Snap.</p>
      </section>
      <aside className="auth-shell__art" aria-label="EventSnap promise">
        <Star className="auth-art__star" size={46} fill="#fffdf6" />
        <Zap className="auth-art__zap" size={58} fill="#f6c84c" />
        <div className="auth-art__eyebrow">{signup ? "YOUR GUESTS ARE WAITING" : "WELCOME BACK"}</div>
        <h2>{signup ? "The best moments should find you." : "Your event’s best frames, waiting for you."}</h2>
        <p>{signup ? "One QR, one selfie, every moment that matters." : "Pick up right where your guests left off."}</p>
        <div className="auth-art__card">
          <Aperture size={26} />
          <span>
            AI matching in under<br />
            <strong>1.4 seconds</strong>
          </span>
        </div>
        <Link to="/" className="auth-art__home">Back to EventSnap</Link>
      </aside>
    </main>
  );
}
