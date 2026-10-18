import { useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowRight, ArrowUpRight, Camera, Check, ChevronDown,
  CloudDownload, Image as GalleryIcon, Menu,
  QrCode, Quote, ScanFace, Star,
  Wand2, X, Zap, Monitor,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import HeroSection from "../components/landing2/HeroSection";

/* ─── Scroll-reveal preset ─── */
const appear = {
  initial: { opacity: 0, y: 36 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-80px" },
  transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] },
};

/* ─── Data ─── */
const promises = [
  ["Recognition at rave speed", "Our vector engine indexes 50,000 photos a minute and returns personal matches in about a second — whether the gallery holds 40 frames or 40,000.", Zap, "bg-violet-600", "01"],
  ["One QR, zero friction", "Attendees scan a code at the door, snap a 1-second selfie in the browser, and they're in. No app download, no sign-up form, no password.", QrCode, "bg-orange-500", "02"],
  ["Full-res, straight from the cloud", "Photographers drop RAW or JPEG folders while shooting. Guests pull print-ready, watermark-free originals the moment they're indexed.", CloudDownload, "bg-emerald-500", "03"],
  ["A live wall that feeds itself", "Beam fresh matches to the main-stage screens as they land. Organizers see social shares multiply up to 4x before the night is over.", Monitor, "bg-pink-500", "04"],
];

const steps = [
  ["Snap a quick selfie", "Guest scans the event QR code and takes one selfie in the browser. Two seconds, nothing to install.", Camera, "bg-cyan-400"],
  ["AI matches the face vector", "EventSnap sweeps every upload against the selfie — new photos included, the moment photographers drop them in.", ScanFace, "bg-violet-600"],
  ["Get your private gallery", "Every frame they appear in, ranked by confidence, ready to download in full resolution and share instantly.", GalleryIcon, "bg-pink-500"],
];

const quotes = [
  ["We handed one QR code out at check-in and never touched a spreadsheet of names again. Guests were downloading photos before dessert.", "Priya M.", "WEDDING PHOTOGRAPHER", "bg-violet-100", "bg-violet-600"],
  ["Around 300 people found their own photos before the after-party even started. Our social mentions quadrupled overnight.", "Arjun K.", "CONFERENCE ORGANIZER", "bg-amber-100", "bg-amber-500"],
  ["The confidence score is what sold our team. No more guessing which blurry face in the crowd is which — it just knows.", "Sneha R.", "COLLEGE FEST LEAD", "bg-emerald-100", "bg-emerald-500"],
];

const faqs = [
  ["Is my face data private?", "Yes. We store only mathematical face embeddings — never raw biometric images. A match key can't be reverse-engineered into a photo of you, and everything is scoped to a single event."],
  ["What happens to my photos after the event?", "The event host controls access and retention. Guests can download their matches any time during the retention window. Everything is deleted at the end of it."],
  ["How big can the photo gallery get?", "Our engine is tested against galleries of 100,000+ frames. Matching time stays under 3 seconds regardless of gallery size."],
  ["How long does setup take on event day?", "About five minutes. Create an event, print or display the QR code, and photographers can start dropping photos from any folder. Attendees self-onboard from their phones."],
  ["Does it work on bad venue Wi-Fi?", "Yes. Selfie capture works on any mobile connection. Photos queue locally and sync when bandwidth is available. We've run it at stadiums on 2G."],
];

const events = ["MUSIC FESTIVALS", "WEDDING GALAS", "TECH CONFERENCES", "MARATHONS & SPORTS", "NIGHTCLUBS", "CORPORATE RETREATS", "EXPOS & TRADE SHOWS", "GRADUATIONS"];

/* ─── Reusable section title ─── */
function Title({ label, labelColor = "text-violet-600", children, text, align = "left" }) {
  return (
    <motion.div {...appear} className={`max-w-2xl ${align === "center" ? "mx-auto text-center" : ""}`}>
      <p className={`mb-4 font-mono-x text-xs font-bold uppercase tracking-[.18em] ${labelColor}`}>{label}</p>
      <h2 className="font-display text-4xl font-extrabold leading-[1.05] tracking-tight sm:text-5xl">{children}</h2>
      {text && (
        <p className={`mt-5 text-base leading-relaxed text-slate-600 sm:text-lg ${align === "center" ? "mx-auto max-w-xl" : ""}`}>
          {text}
        </p>
      )}
    </motion.div>
  );
}

/* ─── Sticky navbar ─── */
function Header({ go }) {
  const [menu, setMenu] = useState(false);
  const navigate = useNavigate();
  const links = [
    ["Manifesto", "manifesto"],
    ["How it works", "how-it-works"],
    ["Live demo", "live-demo"],
    ["Pricing", "pricing"],
    ["FAQ", "faq"],
  ];

  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-slate-200/70 bg-[#FFFDF6]/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-8 lg:px-16">
        {/* Brand */}
        <button onClick={() => go("top")} className="flex items-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-violet-600 ink-border hard-shadow-sm">
            <ScanFace className="h-5 w-5 text-white" />
          </span>
          <span className="font-display text-xl font-extrabold">
            Event<span className="text-violet-600">Snap</span>
          </span>
          <span className="hidden rounded-full bg-amber-300 px-2 py-0.5 font-mono-x text-[10px] font-bold ink-border sm:inline">
            AI V2.0
          </span>
        </button>

        {/* Desktop nav */}
        <nav className="hidden gap-7 lg:flex">
          {links.map(([name, id]) => (
            <button
              key={id}
              onClick={() => go(id)}
              className="group relative text-sm font-semibold text-slate-700 hover:text-slate-900"
            >
              {name}
              <span className="absolute -bottom-1 left-0 h-0.5 w-0 bg-violet-600 transition-all group-hover:w-full" />
            </button>
          ))}
        </nav>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <button onClick={() => navigate("/login")} className="hidden text-sm font-bold sm:block">
            Sign in
          </button>
          <button
            onClick={() => navigate("/signup")}
            className="hidden items-center gap-1.5 rounded-full bg-slate-900 px-5 py-2.5 text-sm font-bold text-white sm:flex ink-border transition-all hover:-translate-y-0.5 hover:shadow-[4px_4px_0_0_#0f172a]"
          >
            Get started <ArrowUpRight className="h-4 w-4" />
          </button>
          <button
            onClick={() => setMenu(!menu)}
            className="grid h-10 w-10 place-items-center rounded-xl bg-white ink-border lg:hidden"
          >
            {menu ? <X /> : <Menu />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menu && (
        <nav className="border-t-2 border-slate-900 bg-white px-6 py-4 lg:hidden">
          {links.map(([name, id]) => (
            <button
              key={id}
              onClick={() => { setMenu(false); go(id); }}
              className="block w-full px-3 py-3 text-left font-display text-lg font-bold hover:bg-violet-50"
            >
              {name}
            </button>
          ))}
          <button
            onClick={() => navigate("/signup")}
            className="mt-2 w-full rounded-full bg-slate-900 py-3 font-bold text-white"
          >
            Get started
          </button>
        </nav>
      )}
    </header>
  );
}

/* ─── Selfie Match Demo Simulator ─── */
function Demo() {
  const [state, setState] = useState("idle");
  const navigate = useNavigate();
  const frames = [
    "photo-1501281668745-f7f57925c3b4",
    "photo-1615870393091-0242c4ba45c5",
    "photo-1658227665335-9cc3da774f99",
    "photo-1511795409834-ef04bbd61622",
    "photo-1492684223066-81342ee5ff30",
    "photo-1506157786151-b8491531f063",
  ];
  const scores = [98.2, 97.5, 96.8, 95.4, 94.9, 93.7];

  const scan = () => {
    setState("scan");
    setTimeout(() => setState("done"), 2400);
  };

  return (
    <section id="live-demo" className="px-4 py-24 sm:px-8 lg:px-16 lg:py-32">
      <div className="mx-auto grid max-w-7xl items-center gap-14 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <Title
            label="TRY IT RIGHT NOW"
            labelColor="text-cyan-600"
            text="This is a simulation of the real attendee flow. Hit the button and watch the engine sweep a 2,048-frame gallery and surface every match with its confidence score."
          >
            Watch a face become<br />a <span className="italic text-violet-600">search query.</span>
          </Title>
          <ul className="mt-8 space-y-3">
            {["No photos leave the device in this demo", "Confidence score on every match", "Works on 40 photos or 40,000"].map((x) => (
              <li className="flex items-center gap-3 text-sm font-semibold" key={x}>
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 ink-border" />{x}
              </li>
            ))}
          </ul>
        </div>

        <motion.div {...appear} className="lg:col-span-7">
          <div className="overflow-hidden rounded-3xl bg-white ink-border hard-shadow-lg">
            {/* Header bar */}
            <div className="flex items-center justify-between border-b-2 border-slate-900 bg-cyan-300 px-6 py-3.5">
              <span className="flex items-center gap-2 font-mono-x text-xs font-bold uppercase tracking-widest">
                <ScanFace className="h-4 w-4" />SELFIE MATCH SIMULATOR
              </span>
              <span className="font-mono-x text-xs font-bold">GALLERY: 2,048 FRAMES</span>
            </div>

            <div className="p-6 sm:p-8">
              {/* Idle */}
              {state === "idle" && (
                <div className="flex flex-col items-center py-10 text-center">
                  <span className="mb-6 grid h-24 w-24 place-items-center rounded-full bg-violet-100 ink-border">
                    <ScanFace className="h-12 w-12 text-violet-600" />
                  </span>
                  <p className="mb-8 font-display text-lg font-bold">One selfie unlocks the whole gallery. Ready?</p>
                  <button
                    onClick={scan}
                    className="flex items-center gap-2 rounded-full bg-violet-600 px-8 py-4 font-display font-bold text-white ink-border hard-shadow transition-all hover:-translate-y-1"
                  >
                    <Wand2 className="h-5 w-5" />Simulate selfie match
                  </button>
                </div>
              )}

              {/* Scanning */}
              {state === "scan" && (
                <div className="py-10">
                  <div className="relative mx-auto mb-8 h-40 max-w-md overflow-hidden rounded-2xl ink-border">
                    <img
                      className="h-full w-full object-cover"
                      src="https://images.unsplash.com/photo-1615870393091-0242c4ba45c5?auto=format&fit=crop&w=800&q=85"
                      alt="Scanning gallery"
                    />
                    <div className="scanner-beam" />
                  </div>
                  <div className="mx-auto max-w-md">
                    <div className="mb-2 flex justify-between font-mono-x text-xs font-bold">
                      <span>SCANNING FACES…</span><span>98%</span>
                    </div>
                    <div className="h-4 overflow-hidden rounded-full bg-slate-100 ink-border">
                      <div className="h-full w-full animate-pulse bg-emerald-500" />
                    </div>
                  </div>
                </div>
              )}

              {/* Done */}
              {state === "done" && (
                <div>
                  <div className="mb-5 flex items-center justify-between">
                    <p className="font-display text-lg font-extrabold">
                      6 matches found{" "}
                      <span className="ml-2 rounded-full bg-emerald-100 px-3 py-1 font-mono-x text-[10px] font-bold text-emerald-700 ink-border">
                        1.2s
                      </span>
                    </p>
                    <button
                      onClick={() => setState("idle")}
                      className="flex items-center gap-1.5 rounded-full bg-white px-4 py-2 text-xs font-bold ink-border hover:bg-amber-300"
                    >
                      <span className="text-base leading-none">↺</span> Run again
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    {frames.map((frame, i) => (
                      <div className="relative aspect-square overflow-hidden rounded-xl ink-border" key={frame}>
                        <img
                          className="h-full w-full object-cover"
                          src={`https://images.unsplash.com/${frame}?auto=format&fit=crop&w=400&q=85`}
                          alt={`Match ${i + 1}`}
                        />
                        <span className="absolute bottom-2 left-2 rounded-full bg-slate-900/90 px-2.5 py-1 font-mono-x text-[10px] font-bold text-emerald-400">
                          {scores[i]}%
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-6 text-center text-sm text-slate-600">
                    Ready to use the real thing?{" "}
                    <button onClick={() => navigate("/signup")} className="font-bold text-violet-600 underline hover:no-underline">
                      Create your first event →
                    </button>
                  </p>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

/* ─── Main Landing Page ─── */
export default function LandingPage() {
  const navigate = useNavigate();
  const [yearly, setYearly] = useState(true);
  const [open, setOpen] = useState(-1);

  const go = (id) =>
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <div className="min-h-screen bg-[#FFFDF6] text-slate-900">
      <Header go={go} />

      <main>
        {/* ── Hero ── */}
        <HeroSection scrollToId={go} />

        {/* ── Marquee strip ── */}
        <section
          className="relative overflow-hidden border-y-2 border-slate-900 bg-violet-600 py-4"
          style={{ transform: "rotate(-1deg)", margin: "0 -12px" }}
        >
          <div className="flex w-max animate-marquee items-center gap-10 pr-10">
            {[...events, ...events].map((x, i) => (
              <span
                key={`${x}${i}`}
                className="flex items-center gap-10 whitespace-nowrap font-display text-lg font-extrabold text-white"
              >
                {x}
                <span className={`text-xl ${i % 3 === 0 ? "text-amber-300" : i % 3 === 1 ? "text-pink-300" : "text-cyan-300"}`}>
                  ✦
                </span>
              </span>
            ))}
          </div>
        </section>

        {/* ── Manifesto / Promises ── */}
        <section id="manifesto" className="px-4 py-24 sm:px-8 lg:px-16 lg:py-32">
          <div className="mx-auto max-w-7xl">
            <Title
              label="THE MANIFESTO"
              labelColor="text-violet-600"
              text="Nobody grades photo software by the pitch deck. They grade it at 11pm, with 4,000 photos and 300 people hammering the gallery at once."
            >
              Four promises we keep,<br />every single <span className="italic text-violet-600">event day.</span>
            </Title>

            <div className="mt-16 grid gap-6 lg:grid-cols-2">
              {promises.map(([title, text, Icon, color, num], i) => (
                <motion.article
                  {...appear}
                  transition={{ delay: i * 0.1, duration: 0.7 }}
                  key={title}
                  className="group relative overflow-hidden rounded-3xl bg-white p-8 ink-border hard-shadow transition-all hover:-translate-y-2 hover:shadow-[10px_10px_0_0_#0f172a]"
                >
                  <span className="pointer-events-none absolute bottom-4 right-6 select-none font-display text-[7rem] font-extrabold leading-none text-slate-100">
                    {num}
                  </span>
                  <span className={`relative mb-6 grid h-14 w-14 place-items-center rounded-2xl ${color} ink-border hard-shadow-sm transition-transform group-hover:-rotate-6 group-hover:scale-110`}>
                    <Icon className="h-7 w-7 text-white" />
                  </span>
                  <h3 className="relative font-display text-2xl font-extrabold">{title}</h3>
                  <p className="relative mt-3 max-w-sm leading-relaxed text-slate-600">{text}</p>
                </motion.article>
              ))}
            </div>
          </div>
        </section>

        {/* ── How it works ── */}
        <section id="how-it-works" className="border-y-2 border-slate-900 bg-[#F3EFE0] px-4 py-24 sm:px-8 lg:px-16 lg:py-32">
          <div className="mx-auto max-w-7xl">
            <Title
              label="THE WHOLE FLOW"
              labelColor="text-cyan-600"
              text="This is the entire attendee experience — nothing to name, nothing to install, nothing to explain at the door."
            >
              Three steps. One code.<br />One <span className="italic text-cyan-500">selfie.</span>
            </Title>

            <div className="relative mt-16 grid gap-6 md:grid-cols-3">
              {/* Connectors */}
              <div className="absolute top-[4.5rem] left-[calc(33.33%+1rem)] right-[calc(33.33%+1rem)] hidden border-t-2 border-dashed border-slate-400 md:block" />
              <div className="absolute top-[4.5rem] left-[calc(66.66%+1rem)] right-4 hidden border-t-2 border-dashed border-slate-400 md:block" />

              {steps.map(([title, text, Icon, color], i) => (
                <motion.article
                  {...appear}
                  transition={{ delay: i * 0.15, duration: 0.7 }}
                  key={title}
                  className={`group relative overflow-hidden rounded-3xl bg-white p-8 ink-border hard-shadow transition-all hover:-translate-y-2 ${i === 1 ? "md:translate-y-6" : ""}`}
                >
                  <div className="mb-8 flex items-center justify-between">
                    <span className={`grid h-14 w-14 place-items-center rounded-2xl ${color} ink-border hard-shadow-sm transition-transform group-hover:rotate-6 group-hover:scale-110`}>
                      <Icon className="h-7 w-7 text-white" />
                    </span>
                    <span className="rounded-full bg-slate-900 px-4 py-1.5 font-mono-x text-[10px] font-bold text-white">
                      STEP 0{i + 1}
                    </span>
                  </div>
                  <h3 className="font-display text-2xl font-extrabold">{title}</h3>
                  <p className="mt-3 leading-relaxed text-slate-600">{text}</p>
                </motion.article>
              ))}
            </div>
          </div>
        </section>

        {/* ── Live Demo ── */}
        <Demo />

        {/* ── Testimonials ── */}
        <section className="border-y-2 border-slate-900 bg-[#F3EFE0] px-4 py-24 sm:px-8 lg:px-16 lg:py-32">
          <div className="mx-auto max-w-7xl">
            <Title label="WHAT ORGANIZERS SAID" labelColor="text-amber-600">
              Not case studies. Just what happened<br />after <span className="italic text-violet-600">one code</span> went out.
            </Title>
            <div className="mt-16 grid gap-8 md:grid-cols-3">
              {quotes.map(([quote, name, role, wash, chip], i) => (
                <motion.figure
                  {...appear}
                  transition={{ delay: i * 0.15, duration: 0.7 }}
                  key={name}
                  className={`relative rounded-3xl ${wash} p-8 ink-border hard-shadow transition-all hover:-translate-y-2 ${i === 1 ? "md:translate-y-6" : ""}`}
                >
                  <span className={`absolute -top-4 left-8 grid h-10 w-10 place-items-center rounded-full ${chip} ink-border`}>
                    <Quote className="h-5 w-5 fill-white text-white" />
                  </span>
                  <blockquote className="mt-4 font-display text-lg font-bold leading-snug">"{quote}"</blockquote>
                  <figcaption className="mt-6 flex items-center gap-3">
                    <span className={`grid h-11 w-11 place-items-center rounded-full ${chip} font-display text-sm font-extrabold text-white ink-border`}>
                      {name[0]}
                    </span>
                    <span>
                      <span className="block font-display text-sm font-extrabold">{name}</span>
                      <span className="block font-mono-x text-[10px] font-semibold tracking-widest text-slate-500">{role}</span>
                    </span>
                  </figcaption>
                </motion.figure>
              ))}
            </div>
          </div>
        </section>

        {/* ── Pricing ── */}
        <section id="pricing" className="px-4 py-24 sm:px-8 lg:px-16 lg:py-32">
          <div className="mx-auto max-w-7xl">
            <div className="flex flex-col items-start justify-between gap-8 md:flex-row md:items-end">
              <Title
                label="PRICING"
                labelColor="text-violet-600"
                text="No per-seat pricing. No per-attendee pricing. Just how many events you're actually running."
              >
                Scales with your events,<br />not your <span className="italic text-orange-500">headcount.</span>
              </Title>

              {/* Billing toggle */}
              <button
                onClick={() => setYearly(!yearly)}
                className="flex shrink-0 items-center gap-3 rounded-full bg-white px-5 py-3 ink-border hard-shadow-sm transition-all hover:-translate-y-0.5"
              >
                <span className={!yearly ? "font-bold" : "text-slate-400"}>Monthly</span>
                <span className="relative h-8 w-16 rounded-full bg-slate-900 p-1 transition-all">
                  <span className={`block h-6 w-6 rounded-full bg-amber-300 transition-transform ${yearly ? "translate-x-8" : ""}`} />
                </span>
                <span className={yearly ? "font-bold" : "text-slate-400"}>
                  Yearly{" "}
                  <b className="ml-1 rounded-full bg-emerald-300 px-2 py-0.5 font-mono-x text-[10px] font-bold ink-border">SAVE 20%</b>
                </span>
              </button>
            </div>

            <div className="mt-16 grid gap-8 lg:grid-cols-3">
              {[
                ["Starter", "For one event at a time.", "Free", null, ["1 active event", "Up to 200 photos", "Standard match speed", "Community support"], false, "Start free"],
                ["Studio", "For recurring events and full seasons.", `$${yearly ? 19 : 24}`, yearly ? "/mo, billed yearly" : "/mo", ["Unlimited events", "Up to 5,000 photos / event", "Priority match speed (<3s)", "Analytics dashboard", "Live photo wall", "Email support"], true, "Start free trial"],
                ["Enterprise", "For venues, agencies and platforms.", "Custom", null, ["Unlimited photos", "Dedicated infrastructure", "SSO & audit logs", "Uptime SLA", "Dedicated account manager"], false, "Talk to us"],
              ].map(([name, tag, price, priceSub, features, featured, cta]) => (
                <article
                  key={name}
                  className={`relative flex flex-col rounded-3xl p-8 ink-border transition-all hover:-translate-y-2 ${featured ? "bg-violet-600 text-white hard-shadow-lg lg:-translate-y-4 lg:hover:-translate-y-6" : "bg-white hard-shadow"}`}
                >
                  {featured && (
                    <span className="absolute -top-5 left-1/2 -translate-x-1/2 flex items-center gap-1.5 whitespace-nowrap rounded-full bg-amber-300 px-4 py-1.5 font-mono-x text-[10px] font-bold text-slate-900 ink-border">
                      ✦ MOST ORGANIZERS PICK THIS
                    </span>
                  )}
                  <h3 className="font-display text-2xl font-extrabold">{name}</h3>
                  <p className={`mt-1 text-sm ${featured ? "text-violet-200" : "text-slate-500"}`}>{tag}</p>
                  <div className="mt-6 flex items-baseline gap-1">
                    <strong className="font-display text-5xl font-extrabold">{price}</strong>
                    {priceSub && <span className={`text-sm ${featured ? "text-violet-200" : "text-slate-500"}`}>{priceSub}</span>}
                  </div>
                  <ul className="mt-7 flex-1 space-y-3">
                    {features.map((x) => (
                      <li className="flex items-center gap-2.5 text-sm font-medium" key={x}>
                        <span className={`grid h-5 w-5 shrink-0 place-items-center rounded-full ${featured ? "bg-amber-300 text-slate-900" : "bg-emerald-400 text-white"}`}>
                          <Check className="h-3 w-3" strokeWidth={3.5} />
                        </span>
                        {x}
                      </li>
                    ))}
                  </ul>
                  <button
                    onClick={() => navigate(`/signup?plan=${name.toLowerCase()}`)}
                    className={`mt-8 w-full rounded-full py-3.5 font-display font-bold ink-border transition-all hover:-translate-y-0.5 ${featured ? "bg-amber-300 text-slate-900 hover:bg-white" : "bg-slate-900 text-white hover:bg-violet-600 hover:text-white"}`}
                  >
                    {cta}
                  </button>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* ── FAQ ── */}
        <section id="faq" className="border-y-2 border-slate-900 bg-[#F3EFE0] px-4 py-24 sm:px-8 lg:px-16 lg:py-32">
          <div className="mx-auto grid max-w-7xl gap-14 lg:grid-cols-12">
            <div className="lg:col-span-5">
              <Title
                label="QUESTIONS"
                labelColor="text-cyan-600"
                text="Privacy, deletion, gallery limits and venue Wi-Fi — the five things every organizer asks before saying yes."
              >
                Asked at every<br /><span className="italic text-pink-500">demo call.</span>
              </Title>
            </div>
            <div className="space-y-3 lg:col-span-7">
              {faqs.map(([question, answer], i) => (
                <article
                  key={question}
                  className={`overflow-hidden rounded-2xl bg-white ink-border hard-shadow-sm transition-all ${open === i ? "bg-violet-50" : ""}`}
                >
                  <button
                    onClick={() => setOpen(open === i ? -1 : i)}
                    className="flex w-full items-center justify-between px-6 py-5 text-left font-display text-lg font-extrabold"
                  >
                    {question}
                    <ChevronDown className={`ml-4 h-5 w-5 shrink-0 transition-transform duration-300 ${open === i ? "rotate-180" : ""}`} />
                  </button>
                  {open === i && (
                    <p className="px-6 pb-5 text-sm leading-relaxed text-slate-600">{answer}</p>
                  )}
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section className="px-4 py-24 sm:px-8 lg:px-16">
          <div
            className="relative mx-auto overflow-hidden rounded-[2.5rem] px-6 py-20 text-center ink-border hard-shadow-lg lg:max-w-7xl"
            style={{ background: "linear-gradient(135deg, #f97316 0%, #fb923c 40%, #ef4444 100%)" }}
          >
            <div className="pointer-events-none absolute -left-16 -top-16 h-64 w-64 rounded-full bg-amber-300/40 blur-3xl" />
            <div className="pointer-events-none absolute -bottom-12 -right-12 h-64 w-64 rounded-full bg-red-400/40 blur-3xl" />
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 16, repeat: Infinity, ease: "linear" }}
              className="absolute right-10 top-10 hidden lg:block"
            >
              <Star className="h-14 w-14 fill-amber-300 text-amber-300" strokeWidth={1.5} />
            </motion.div>
            <p className="relative mb-5 font-mono-x text-xs font-bold tracking-[.18em] text-white/90">YOUR MOVE, ORGANIZER</p>
            <h2 className="relative font-display text-4xl font-extrabold text-white sm:text-5xl lg:text-6xl">
              Ready to hand every guest<br />
              <span className="italic text-amber-200">their own</span> highlight reel?
            </h2>
            <p className="relative mx-auto mt-6 max-w-xl text-orange-50">
              Spin up your first event in five minutes. Free plan, no card, one QR code between you and a crowd that finds itself.
            </p>
            <div className="relative mt-10 flex flex-wrap justify-center gap-4">
              <button
                onClick={() => navigate("/signup")}
                className="flex items-center gap-2 rounded-full bg-slate-900 px-8 py-4 font-display font-bold text-white ink-border transition-all hover:-translate-y-1 hover:shadow-[6px_6px_0_0_rgba(0,0,0,0.4)]"
              >
                Start free today <ArrowRight className="h-5 w-5" />
              </button>
              <button
                onClick={() => go("live-demo")}
                className="rounded-full bg-white px-8 py-4 font-display font-bold ink-border transition-all hover:-translate-y-1 hover:bg-amber-300"
              >
                Replay the demo
              </button>
            </div>
          </div>
        </section>
      </main>

      {/* ── Footer ── */}
      <footer className="relative overflow-hidden bg-slate-900 px-4 pt-16 text-white sm:px-8 lg:px-16">
        <div className="mx-auto grid max-w-7xl gap-12 pb-12 md:grid-cols-[1.4fr_2fr]">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="grid h-9 w-9 place-items-center rounded-xl bg-violet-600 ink-border">
                <ScanFace className="h-5 w-5 text-white" />
              </span>
              <p className="font-display text-2xl font-extrabold">Event<span className="text-violet-400">Snap</span></p>
            </div>
            <p className="mt-4 max-w-sm text-sm leading-relaxed text-slate-400">
              Face matching for event photography. One selfie in, every memory out — private by design, fast by default.
            </p>
            <p className="mt-5 flex items-center gap-2 font-mono-x text-[10px] tracking-widest text-emerald-400">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              ALL SYSTEMS OPERATIONAL
            </p>
          </div>

          <div className="grid grid-cols-3 gap-6 text-sm">
            {[
              ["PRODUCT", "How it works", "Live demo", "Pricing", "FAQ"],
              ["COMPANY", "About", "Careers", "Press kit", "Contact"],
              ["LEGAL", "Privacy", "Biometric policy", "Terms", "DPA"],
            ].map(([heading, ...links]) => (
              <div key={heading}>
                <p className="mb-4 font-mono-x text-[10px] tracking-widest text-slate-500">{heading}</p>
                {links.map((link) => (
                  <button className="mb-3 block text-left text-slate-300 hover:text-white transition-colors" key={link}>
                    {link}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Watermark */}
        <div className="pointer-events-none select-none overflow-hidden">
          <p
            className="text-center font-display font-extrabold leading-none tracking-tighter text-slate-800"
            style={{ fontSize: "clamp(60px, 12vw, 160px)" }}
          >
            EVENT SNAP
          </p>
        </div>

        {/* Bottom bar */}
        <div className="mx-auto flex max-w-7xl items-center justify-between border-t border-slate-800 py-6">
          <p className="font-mono-x text-[10px] tracking-widest text-slate-500">
            © 2026 EVENTSNAP INC. MADE FOR THE FRONT ROW.
          </p>
          <div className="flex items-center gap-3">
            {[
              {
                label: "Twitter",
                icon: <svg viewBox="0 0 24 24" className="h-4 w-4 fill-current"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.743l7.73-8.835L1.254 2.25H8.08l4.259 5.63L18.244 2.25zm-1.161 17.52h1.833L7.084 4.126H5.117L17.083 19.77z" /></svg>,
              },
              {
                label: "Instagram",
                icon: <svg viewBox="0 0 24 24" className="h-4 w-4 fill-none stroke-current" strokeWidth="2"><rect x="2" y="2" width="20" height="20" rx="5" /><circle cx="12" cy="12" r="4" /><circle cx="17.5" cy="6.5" r="1" fill="currentColor" strokeWidth="0" /></svg>,
              },
              {
                label: "LinkedIn",
                icon: <svg viewBox="0 0 24 24" className="h-4 w-4 fill-current"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" /></svg>,
              },
            ].map(({ label, icon }) => (
              <button
                key={label}
                aria-label={label}
                className="grid h-9 w-9 place-items-center rounded-full border border-slate-700 text-slate-400 transition-colors hover:border-slate-500 hover:text-white"
              >
                {icon}
              </button>
            ))}
          </div>
        </div>
      </footer>
    </div>
  );
}
