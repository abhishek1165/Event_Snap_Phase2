import React from "react";
import { cn } from "@/lib/utils";
import Logo from "@/components/Logo";

/* ════════════════════════════════════════════════════════════════════
   AppShell — shared chrome for the authenticated/attendee app pages
   (Dashboard, EventDetails, SelfieCapture, PhotoGallery).

   Replaces the hand-rolled per-page headers with ONE consistent layout:
   glass top-nav + layered ambient background. Uses min-h-[100dvh]
   (never h-screen) per design-taste §3.E viewport stability.
   ════════════════════════════════════════════════════════════════════ */

export function AmbientBackground() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden"
    >
      {/* deep base */}
      <div className="absolute inset-0 bg-ink" />
      {/* two diffused, cool-tinted glows (Linear-style ambient) */}
      <div className="absolute -top-48 -left-32 h-[520px] w-[520px] rounded-full bg-iris/[0.07] blur-[120px]" />
      <div className="absolute top-1/3 -right-24 h-[420px] w-[420px] rounded-full bg-match/[0.05] blur-[120px]" />
      {/* faint top vignette to anchor the nav */}
      <div className="absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-iris/[0.06] to-transparent" />
    </div>
  );
}

export function AppHeader({
  children,
  right,
  showLogo = true,
  onLogoClick,
  className,
}) {
  return (
    <header
      className={cn(
        "sticky top-0 z-40 border-b border-frame-soft",
        // glass — only here, where it signals background dismissal (§blur-purpose)
        "bg-ink/70 backdrop-blur-xl"
      )}
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5 sm:px-6 lg:px-8">
        <div className="flex items-center gap-6">
          {showLogo && <Logo size={26} onClick={onLogoClick} />}
          {children}
        </div>
        <div className="flex items-center gap-3">{right}</div>
      </div>
    </header>
  );
}

export function AppShell({ header, children, className }) {
  return (
    <div className="relative min-h-[100dvh] text-text">
      <AmbientBackground />
      {header}
      <main className={cn("relative z-10", className)}>{children}</main>
    </div>
  );
}

export default AppShell;
