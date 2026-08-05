import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(useGSAP, ScrollTrigger);

const prefersReduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * useReveal — scroll-triggered reveal for any [data-reveal] element
 * inside the returned scope ref. One hook per page/section, scoped,
 * auto-cleaned. Respects prefers-reduced-motion.
 *
 *   const scope = useReveal();
 *   <div ref={scope}> … <div data-reveal>…</div> … </div>
 *
 * Optional opts: { y, duration, start, stagger }
 */
export function useReveal(opts = {}) {
  const scope = useRef(null);
  const { y = 24, duration = 0.7, start = "top 87%", stagger } = opts;

  useGSAP(
    () => {
      if (prefersReduced()) return;
      const els = gsap.utils.toArray("[data-reveal]");
      if (stagger) {
        // group reveal with a shared stagger (parent has [data-reveal-group])
        gsap.utils.toArray("[data-reveal-group]").forEach((group) => {
          const kids = group.querySelectorAll("[data-reveal]");
          if (kids.length) {
            gsap.from(kids, {
              opacity: 0,
              y,
              duration,
              stagger,
              ease: "power3.out",
              scrollTrigger: { trigger: group, start },
            });
          }
        });
        // also reveal any standalone [data-reveal] not inside a group
        gsap.utils.toArray("[data-reveal]").forEach((el) => {
          if (el.closest("[data-reveal-group]")) return;
          gsap.fromTo(
            el,
            { opacity: 0, y },
            {
              opacity: 1,
              y: 0,
              duration,
              ease: "power3.out",
              scrollTrigger: { trigger: el, start },
            }
          );
        });
      } else {
        els.forEach((el) =>
          gsap.fromTo(
            el,
            { opacity: 0, y },
            {
              opacity: 1,
              y: 0,
              duration,
              ease: "power3.out",
              scrollTrigger: { trigger: el, start },
            }
          )
        );
      }
    },
    { scope }
  );

  return scope;
}

export default useReveal;
