import { useEffect, useRef } from "react";
import gsap from "gsap";

/**
 * useCountUp — animate a number from 0 → target on mount, writing into the
 * textContent of the element ref passed in. Respects prefers-reduced-motion
 * (snaps instantly to final value).
 *
 *   const ref = useCountUp(98.7, { decimals: 1 });
 *   <span ref={ref}>0</span>
 */
export function useCountUp(target, opts = {}) {
  const elRef = useRef(null);
  const { decimals = 0, duration = 1.6, delay = 0.3, prefix = "", suffix = "" } = opts;

  useEffect(() => {
    if (!elRef.current) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reduced) {
      elRef.current.textContent = prefix + target.toFixed(decimals) + suffix;
      return;
    }

    const proxy = { v: 0 };
    const tween = gsap.to(proxy, {
      v: target,
      duration,
      delay,
      ease: "power2.out",
      onUpdate: () => {
        if (elRef.current)
          elRef.current.textContent =
            prefix + proxy.v.toFixed(decimals) + suffix;
      },
    });
    return () => tween.kill();
  }, [target, decimals, duration, delay, prefix, suffix]);

  return elRef;
}

export default useCountUp;
