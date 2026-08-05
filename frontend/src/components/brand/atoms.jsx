import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";
import { cn } from "@/lib/utils";

/* ════════════════════════════════════════════════════════════════════
   Event Snap — premium brand atoms (used by app pages)
   Built with CVA per tailwind-design-system skill. Uses semantic tokens
   (bg-surface / text-iris / etc.) instead of hardcoded hex.
   ════════════════════════════════════════════════════════════════════ */

const focusRing =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-iris-2 focus-visible:ring-offset-2 focus-visible:ring-offset-ink";

/* ── Button ─────────────────────────────────────────────────────── */
export const buttonVariants = cva(
  // base — transitions only color/opacity/transform, never `all`
  cn(
    "relative inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl",
    "font-medium tracking-tight transition-[transform,background-color,border-color,box-shadow,opacity]",
    "duration-200 ease-out active:scale-[.98]",
    "disabled:pointer-events-none disabled:opacity-50",
    focusRing,
    "[&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0"
  ),
  {
    variants: {
      variant: {
        // primary — solid brand violet, tinted edge shadow (NOT a neon glow).
        // Per design-taste §4.2 LILA RULE: brand is locked violet, executed
        // with intent + restraint, desaturated lift not slop.
        primary:
          "bg-iris text-white shadow-[0_1px_0_0_rgba(255,255,255,.18)_inset,0_8px_24px_-12px_rgba(109,94,245,.55)] hover:bg-iris-2 hover:-translate-y-0.5",
        // gradient — reserved for ONE hero moment only (not everywhere)
        gradient:
          "bg-brand-gradient text-white shadow-[0_1px_0_0_rgba(255,255,255,.18)_inset,0_8px_24px_-12px_rgba(109,94,245,.45)] hover:-translate-y-0.5",
        // solid match (success / "download")
        match:
          "bg-match text-ink font-semibold shadow-[0_8px_24px_-12px_rgba(52,211,153,.5)] hover:-translate-y-0.5",
        // ghost — hairline border that lights up on hover
        ghost:
          "border border-frame bg-transparent text-text hover:border-iris-2 hover:bg-surface-2/60",
        // subtle — surface fill
        subtle:
          "bg-surface-2 text-text hover:bg-surface-3 border border-frame-soft",
        // outline (lighter than ghost, for dark-on-dark)
        outline:
          "border border-frame-soft bg-white/[0.02] text-text hover:bg-white/[0.05]",
        // link
        link: "text-iris-2 underline-offset-4 hover:underline h-auto p-0",
      },
      size: {
        sm: "h-9 px-3.5 text-[0.85rem]",
        md: "h-11 px-5 text-sm",
        lg: "h-12 px-6 text-[0.95rem]",
        xl: "h-14 px-8 text-base",
        icon: "h-11 w-11",
        "icon-sm": "h-9 w-9",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  }
);

export const Button = React.forwardRef(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        className={cn(buttonVariants({ variant, size, className }))}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

/* ── Card family (compound, tailwind-design-system pattern) ─────── */
export const Card = React.forwardRef(({ className, glow = false, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "relative rounded-2xl border border-frame-soft bg-surface text-text",
      "shadow-[0_1px_0_0_rgba(255,255,255,.04)_inset,0_24px_48px_-24px_rgba(0,0,0,.6)]",
      glow && "before:absolute before:inset-0 before:rounded-2xl before:bg-edge-light before:[mask:linear-gradient(#fff_0_0)_content-box,linear-gradient(#fff_0_0)] before:[mask-composite:exclude] before:p-px before:pointer-events-none",
      className
    )}
    {...props}
  />
));
Card.displayName = "Card";

export const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col gap-1.5 p-6", className)} {...props} />
));
CardHeader.displayName = "CardHeader";

export const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn("text-lg font-semibold tracking-tight", className)}
    {...props}
  />
));
CardTitle.displayName = "CardTitle";

export const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p ref={ref} className={cn("text-sm text-text-dim", className)} {...props} />
));
CardDescription.displayName = "CardDescription";

export const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

export const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
));
CardFooter.displayName = "CardFooter";

/* ── Container (responsive max-width) ───────────────────────────── */
export function Container({ className, size = "xl", ...props }) {
  const max = {
    sm: "max-w-screen-sm",
    md: "max-w-screen-md",
    lg: "max-w-screen-lg",
    xl: "max-w-7xl",
    full: "max-w-full",
  }[size];
  return <div className={cn("mx-auto w-full px-5 sm:px-6 lg:px-8", max, className)} {...props} />;
}

/* ── Eyebrow label ────────────────────────────────────────────────
   NOTE (design-taste §4.7 EYEBROW RESTRAINT): use sparingly, max 1 per
   3 sections. No decorative dot prefix by default (§9.F). */
export function Eyebrow({ className, children, ...props }) {
  return (
    <span
      className={cn(
        "inline-block font-mono text-[0.7rem] uppercase tracking-[0.16em] text-iris-2",
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}

/* ── Badge (status pills) ───────────────────────────────────────── */
const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.7rem] font-semibold tracking-wide",
  {
    variants: {
      tone: {
        iris:  "bg-iris/[.12] text-iris-2",
        match: "bg-match/10 text-match",
        amber: "bg-memory/10 text-memory",
        muted: "bg-white/[.05] text-text-dim",
        danger:"bg-destructive/10 text-destructive-foreground",
      },
    },
    defaultVariants: { tone: "muted" },
  }
);
export function Badge({ className, tone, ...props }) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}

/* ── Input (with error + aria pattern) ──────────────────────────── */
export const Input = React.forwardRef(({ className, error, ...props }, ref) => (
  <div className="relative">
    <input
      ref={ref}
      aria-invalid={!!error || undefined}
      className={cn(
        "flex h-12 w-full rounded-xl border bg-surface-2 px-4 text-[0.95rem] text-text",
        "placeholder:text-text-faint transition-[border-color,box-shadow]",
        "focus-visible:outline-none focus-visible:border-iris-2 focus-visible:ring-2 focus-visible:ring-iris-2/40",
        error ? "border-destructive" : "border-frame",
        className
      )}
      {...props}
    />
    {error && (
      <p className="mt-1.5 text-xs text-destructive-foreground" role="alert">
        {error}
      </p>
    )}
  </div>
));
Input.displayName = "Input";

/* ── Field label ────────────────────────────────────────────────── */
export function Label({ className, children, ...props }) {
  return (
    <label
      className={cn(
        "block text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-text-dim mb-2",
        className
      )}
      {...props}
    >
      {children}
    </label>
  );
}
