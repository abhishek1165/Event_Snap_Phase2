import { useRef } from "react";
import { motion, useMotionValue, useScroll, useSpring, useTransform } from "framer-motion";
import { ArrowRight, Play, ScanFace, Sparkles, Star, Zap } from "lucide-react";
import { useNavigate } from "react-router-dom";

const photos = [
  ["https://images.unsplash.com/photo-1501281668745-f7f57925c3b4?auto=format&fit=crop&w=300&q=85", "98.2%"],
  ["https://images.unsplash.com/photo-1658227665335-9cc3da774f99?auto=format&fit=crop&w=300&q=85", null],
  ["https://images.unsplash.com/photo-1652226647060-775569c1dd4e?auto=format&fit=crop&w=300&q=85", "96.7%"],
  ["https://images.unsplash.com/photo-1615870393091-0242c4ba45c5?auto=format&fit=crop&w=300&q=85", null],
  ["https://images.unsplash.com/photo-1511795409834-ef04bbd61622?auto=format&fit=crop&w=300&q=85", null],
  ["https://images.unsplash.com/photo-1501281668745-f7f57925c3b4?auto=format&fit=crop&w=301&q=85", "94.1%"],
];

function MatchCard() {
  const card = useRef(null);
  const rx = useMotionValue(0);
  const ry = useMotionValue(0);
  const rotateX = useSpring(rx, { stiffness: 120, damping: 18 });
  const rotateY = useSpring(ry, { stiffness: 120, damping: 18 });

  const move = (e) => {
    const rect = card.current.getBoundingClientRect();
    ry.set(((e.clientX - rect.left) / rect.width - 0.5) * 10);
    rx.set(-((e.clientY - rect.top) / rect.height - 0.5) * 8);
  };

  return (
    <div style={{ perspective: 1200 }}>
      <motion.div
        ref={card}
        onMouseMove={move}
        onMouseLeave={() => { rx.set(0); ry.set(0); }}
        initial={{ opacity: 0, y: 60, rotate: 2 }}
        animate={{ opacity: 1, y: 0, rotate: 0 }}
        transition={{ delay: 0.7, duration: 1, ease: [0.22, 1, 0.36, 1] }}
        style={{ rotateX, rotateY, transformStyle: "preserve-3d" }}
        className="relative overflow-hidden rounded-3xl bg-white ink-border hard-shadow-lg"
      >
        {/* Card header */}
        <div className="flex items-center justify-between border-b-2 border-slate-900 bg-amber-300 px-5 py-3">
          <span className="flex items-center gap-2 font-mono-x text-xs font-bold uppercase tracking-widest">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse-dot" />
            Live match
          </span>
          <span className="font-mono-x text-xs font-bold text-slate-700">2,048 FRAMES</span>
        </div>

        {/* Photo grid */}
        <div className="relative grid grid-cols-3 gap-2 p-4">
          {photos.map(([src, confidence], i) => (
            <div className="relative aspect-square overflow-hidden rounded-xl ink-border" key={`${src}-${i}`}>
              <img src={src} alt="Event frame" className="h-full w-full object-cover" loading="lazy" />
              {confidence && (
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 1.6 + i * 0.2, type: "spring", stiffness: 300, damping: 15 }}
                  className="absolute right-1 top-1 rounded-full bg-violet-600 px-2 py-0.5 font-mono-x text-[10px] font-bold text-white"
                >
                  {confidence}
                </motion.span>
              )}
            </div>
          ))}
          {/* Sweep beam */}
          <motion.div
            animate={{ top: ["-12%", "104%", "-12%"] }}
            transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut" }}
            className="pointer-events-none absolute left-0 right-0 h-10"
            style={{
              background: "linear-gradient(180deg,transparent,rgba(124,58,237,.35),rgba(6,182,212,.35),transparent)",
              filter: "blur(2px)",
            }}
          />
        </div>

        {/* Stats bar */}
        <div className="grid grid-cols-3 divide-x-2 divide-slate-900 border-t-2 border-slate-900">
          {[["SCANNED", "2,048"], ["MATCHED", "37"], ["CONFIDENCE", "98.2%"]].map(([label, value]) => (
            <div className="px-4 py-3" key={label}>
              <p className="font-mono-x text-[10px] font-bold text-slate-500">{label}</p>
              <p className="font-display text-lg font-extrabold">{value}</p>
            </div>
          ))}
        </div>

        {/* Match badge */}
        <motion.div
          initial={{ opacity: 0, scale: 0.6, rotate: -8 }}
          animate={{ opacity: 1, scale: 1, rotate: -6 }}
          transition={{ delay: 1.9, type: "spring", stiffness: 200, damping: 14 }}
          className="absolute -right-2 -top-4 rounded-full bg-emerald-400 px-4 py-2 font-mono-x text-xs font-bold ink-border hard-shadow-sm"
        >
          MATCH FOUND
        </motion.div>
      </motion.div>
    </div>
  );
}

const reveal = {
  hidden: { y: "115%" },
  visible: (i) => ({
    y: 0,
    transition: { delay: 0.35 + i * 0.14, duration: 0.9, ease: [0.22, 1, 0.36, 1] },
  }),
};

const fade = {
  hidden: { opacity: 0, y: 24 },
  visible: (delay) => ({
    opacity: 1, y: 0,
    transition: { delay, duration: 0.7, ease: [0.22, 1, 0.36, 1] },
  }),
};

export default function HeroSection({ scrollToId }) {
  const navigate = useNavigate();
  const hero = useRef(null);
  const { scrollYProgress } = useScroll({ target: hero, offset: ["start start", "end start"] });
  const cardY   = useTransform(scrollYProgress, [0, 1], [0, 90]);
  const shapeA  = useTransform(scrollYProgress, [0, 1], [0, -140]);
  const shapeB  = useTransform(scrollYProgress, [0, 1], [0, -70]);

  const headLines = [
    <>Every photo</>,
    <><span className="relative inline-block"><span className="relative z-10">you&apos;re&nbsp;in,</span><span className="absolute inset-x-0 bottom-1 z-0 h-4 -rotate-1 rounded-sm bg-pink-300 sm:h-6" /></span></>,
    <>found in a <span className="italic text-violet-600">flash.</span></>,
  ];

  return (
    <section id="top" ref={hero} className="relative overflow-hidden px-4 pb-20 pt-32 sm:px-8 lg:px-16 lg:pt-40">
      {/* Atmospheric blobs */}
      <motion.div style={{ y: shapeA }} className="pointer-events-none absolute -left-16 top-24 h-56 w-56 rounded-full bg-pink-300/70 blur-2xl" />
      <motion.div style={{ y: shapeB }} className="pointer-events-none absolute -right-10 top-64 h-72 w-72 rounded-full bg-cyan-200/80 blur-3xl" />
      <motion.div
        style={{ y: shapeA, "--float-rotate": "12deg" }}
        className="pointer-events-none absolute right-[38%] top-28 hidden animate-float lg:block"
      >
        <Star className="h-10 w-10 fill-amber-300 text-slate-900" strokeWidth={1.5} />
      </motion.div>

      <div className="mx-auto grid max-w-7xl items-center gap-14 lg:grid-cols-12">
        {/* Copy */}
        <div className="lg:col-span-7">
          {/* Pills */}
          <motion.div variants={fade} initial="hidden" animate="visible" custom={0.15} className="mb-7 flex flex-wrap items-center gap-3">
            <span className="flex -rotate-2 items-center gap-1.5 rounded-full bg-emerald-300 px-4 py-1.5 text-xs font-bold ink-border hard-shadow-sm">
              <Sparkles className="h-3.5 w-3.5" />99.8% match accuracy
            </span>
            <span className="flex rotate-[1.5deg] items-center gap-1.5 rounded-full bg-white px-4 py-1.5 text-xs font-bold ink-border hard-shadow-sm">
              <Zap className="h-3.5 w-3.5 fill-amber-400 text-slate-900" />1 selfie is all it takes
            </span>
          </motion.div>

          {/* Headline */}
          <h1 className="font-display text-4xl font-extrabold leading-[1.02] tracking-tight sm:text-6xl lg:text-7xl">
            {headLines.map((line, i) => (
              <span className="block overflow-hidden pb-1" key={i}>
                <motion.span variants={reveal} initial="hidden" animate="visible" custom={i} className="block">
                  {line}
                </motion.span>
              </span>
            ))}
          </h1>

          {/* Sub */}
          <motion.p variants={fade} initial="hidden" animate="visible" custom={0.85} className="mt-6 max-w-xl text-base leading-relaxed text-slate-600 sm:text-lg">
            EventSnap reads one selfie and hands you every frame you appear in — matched, ranked and ready to download before the after-party starts. No tags. No digging. No "send me the link when it's up."
          </motion.p>

          {/* CTAs */}
          <motion.div variants={fade} initial="hidden" animate="visible" custom={1} className="mt-9 flex flex-wrap items-center gap-4">
            <button
              onClick={() => scrollToId("live-demo")}
              className="group flex items-center gap-2 rounded-full bg-violet-600 px-7 py-4 font-display text-base font-bold text-white ink-border hard-shadow transition-all duration-200 hover:-translate-y-1 hover:shadow-[8px_8px_0_0_#0f172a]"
            >
              Try the live demo <ArrowRight className="h-5 w-5 transition-transform duration-200 group-hover:translate-x-1" />
            </button>
            <button
              onClick={() => navigate("/attendjoin")}
              className="flex items-center gap-2 rounded-full bg-white px-7 py-4 font-display text-base font-bold ink-border transition-all duration-200 hover:-translate-y-1 hover:bg-amber-300"
            >
              <ScanFace className="h-4 w-4 text-slate-900" />Find my photo
            </button>
          </motion.div>

          {/* Stats strip */}
          <motion.dl variants={fade} initial="hidden" animate="visible" custom={1.15} className="mt-12 grid max-w-lg grid-cols-3 gap-6 border-t-2 border-dashed border-slate-300 pt-7">
            {[["50K+", "photos scanned"], ["1.2s", "median match"], ["99.8%", "accuracy"]].map(([value, label]) => (
              <div key={label}>
                <dt className="font-display text-2xl font-extrabold sm:text-3xl">{value}</dt>
                <dd className="mt-1 font-mono-x text-[10px] font-semibold uppercase tracking-widest text-slate-500">{label}</dd>
              </div>
            ))}
          </motion.dl>
        </div>

        {/* Match card with parallax */}
        <motion.div style={{ y: cardY }} className="lg:col-span-5">
          <MatchCard />
        </motion.div>
      </div>
    </section>
  );
}
