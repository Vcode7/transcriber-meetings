import React, { useEffect, useLayoutEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Mic,
  Zap,
  Users,
  Wand2,
  FileText,
  Layers,
  Star,

  MessageSquare,
  Sparkles,
} from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";
import { cn } from "@/lib/utils";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

// ─── 1. Squiggle SVG Filter ────────────────────────────────────────────────
function SquiggleFilter() {
  return (
    <svg
      aria-hidden="true"
      width="0"
      height="0"
      style={{ position: "absolute", width: 0, height: 0 }}
    >
      <defs>
        <filter id="lp-squiggle">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.02"
            numOctaves="3"
            result="noise"
            seed="2"
          />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="2.2" />
        </filter>
        <filter id="lp-squiggle-soft">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.015"
            numOctaves="2"
            result="noise"
            seed="5"
          />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="1.2" />
        </filter>
      </defs>
    </svg>
  );
}

// ─── 2. Graph Paper Background ─────────────────────────────────────────────
function GraphPaper({ className }: { className?: string }) {
  return (
    <div
      className={cn("absolute inset-0 pointer-events-none", className)}
      style={{
        backgroundImage: `
          linear-gradient(hsl(222 47% 11% / 0.06) 1px, transparent 1px),
          linear-gradient(90deg, hsl(222 47% 11% / 0.06) 1px, transparent 1px),
          linear-gradient(hsl(222 47% 11% / 0.12) 1px, transparent 1px),
          linear-gradient(90deg, hsl(222 47% 11% / 0.12) 1px, transparent 1px)
        `,
        backgroundSize: "16px 16px, 16px 16px, 80px 80px, 80px 80px",
      }}
    />
  );
}

// ─── 3. Sketch Button ──────────────────────────────────────────────────────
function SketchButton({
  children,
  className,
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost";
}) {
  return (
    <button
      className={cn(
        "btn",
        variant === "primary" ? "btn-primary" : "btn-ghost",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}

// ─── 4. Sticky Note ───────────────────────────────────────────────────────
function StickyNote({
  children,
  color = "bg-yellow-200",
  rotate = 0,
  className,
}: {
  children: React.ReactNode;
  color?: string;
  rotate?: number;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "relative p-5 shadow-sketch border-2 border-ink font-hand hover:-rotate-1 transition-transform duration-200 tape",
        color,
        className
      )}
      style={{
        transform: `rotate(${rotate}deg)`,
        borderRadius: "10px 18px 12px 16px / 16px 12px 18px 10px",
      }}
    >
      {children}
    </div>
  );
}

// ─── 5. Tape Marquee ──────────────────────────────────────────────────────
function TapeMarquee() {
  const items = Array.from({ length: 14 });
  const marqueeItems = [
    "VoiceSum AI",
    "★",
    "Speaker Diarization",
    "★",
    "Smart Transcription",
    "★",
    "AI Insights",
    "★",
  ];
  return (
    <div
      className="relative overflow-hidden border-y-[2.5px] border-ink bg-accent/95 py-3"
      style={{ transform: "rotate(-1deg)", margin: "0 -2rem" }}
    >
      <div className="flex animate-marquee whitespace-nowrap">
        {[...items, ...items].map((_, i) => (
          <span
            key={i}
            className="mx-6 inline-flex items-center gap-4 font-display text-3xl font-bold text-accent-foreground"
          >
            <Mic size={22} />
            <span>{marqueeItems[i % marqueeItems.length]}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── 6. Hero Section ──────────────────────────────────────────────────────
function Hero() {
  return (
    <section className="relative min-h-[92vh] flex flex-col items-center justify-center px-6 text-center overflow-hidden">
      <GraphPaper />

      {/* Floating decorations */}
      <div
        className="absolute top-20 left-12 w-20 h-20 border-2 border-ink/30 rounded-full animate-float"
        style={{ filter: "url(#lp-squiggle)" }}
      />
      <div
        className="absolute bottom-24 right-16 w-12 h-12 border-2 border-accent/50 rounded-full animate-float"
        style={{ animationDelay: "1s", filter: "url(#lp-squiggle)" }}
      />
      <div
        className="absolute top-40 right-24 w-8 h-8 bg-sticky-yellow border-2 border-ink/40 rotate-12"
        style={{ filter: "url(#lp-squiggle-soft)" }}
      />

      {/* Badge */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="mb-6 mt-10"
      >
        <span className="badge badge-yellow">
          <Sparkles size={12} />
          Prototype v2.0 — AI Powered
        </span>
      </motion.div>

      {/* Headline */}
      <motion.h1
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.1 }}
        className="font-display text-6xl md:text-8xl font-bold text-ink leading-tight mb-4 max-w-4xl"
        style={{ filter: "url(#lp-squiggle-soft)" }}
      >
        Transcribe{" "}
        <span className="text-accent scribble-underline">Your Meetings</span>
        <br />
        Into Insights.
      </motion.h1>

      {/* Sub */}
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.25 }}
        className="font-hand text-xl text-pencil max-w-xl mb-8 leading-relaxed"
      >
        VoiceSum turns your conversations into structured transcripts, speaker
        maps, and AI-generated summaries. Built for the humans who think in
        spoken words.
      </motion.p>

      {/* CTA Buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        className="flex flex-wrap items-center justify-center gap-4 mb-12"
      >
        <Link to="/signup">
          <SketchButton variant="primary" className="flex items-center gap-2 px-6 py-3 text-base">
            Start Recording <ArrowRight size={16} />
          </SketchButton>
        </Link>
        <Link to="/login">
          <SketchButton variant="ghost" className="flex items-center gap-2 px-6 py-3 text-base">
            Sign In
          </SketchButton>
        </Link>
      </motion.div>

      {/* Hero UI Mockup */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.5 }}
        className="relative w-full max-w-3xl mx-auto"
      >
        <div
          className="sketch-border bg-card p-6 shadow-sketch-lg"
          style={{ filter: "url(#lp-squiggle-soft)" }}
        >
          {/* Fake waveform */}
          <div className="flex items-end justify-center gap-1 h-20 mb-4">
            {Array.from({ length: 40 }).map((_, i) => (
              <div
                key={i}
                className="bg-accent/70 rounded-sm"
                style={{
                  width: 6,
                  height: `${20 + Math.sin(i * 0.8) * 30 + Math.random() * 20}%`,
                  animation: `float-y ${1.5 + (i % 5) * 0.3}s ease-in-out infinite`,
                  animationDelay: `${i * 0.04}s`,
                }}
              />
            ))}
          </div>
          {/* Fake transcript lines */}
          <div className="space-y-2">
            {[
              { speaker: "Speaker A", text: "Let's start with the project overview..." },
              { speaker: "Speaker B", text: "Sure, I'll share my screen for the slides." },
              { speaker: "Speaker A", text: "Perfect. The key deliverable is..." },
            ].map((seg, i) => (
              <div
                key={i}
                className="flex gap-3 p-2 rounded-lg"
                style={{
                  background:
                    i % 2 === 0
                      ? "hsl(var(--sticky-yellow) / .25)"
                      : "hsl(var(--sticky-blue) / .25)",
                  borderRadius: "10px 14px 12px 10px / 12px 10px 14px 12px",
                }}
              >
                <span className="font-blueprint text-xs font-bold text-accent whitespace-nowrap">
                  {seg.speaker}
                </span>
                <span className="font-hand text-sm text-ink">{seg.text}</span>
              </div>
            ))}
          </div>
        </div>
        {/* Floating sticky note */}
        <div
          className="absolute -top-6 -right-4 bg-sticky-pink border-2 border-ink p-2 text-xs font-bold font-hand rotate-6 shadow-sketch"
          style={{ borderRadius: "8px 14px 10px 12px / 12px 10px 14px 8px" }}
        >
          <Mic size={14} className="inline mr-1 text-accent" />
          Live!
        </div>
      </motion.div>
    </section>
  );
}

// ─── 7. Feature Board ─────────────────────────────────────────────────────
function FeatureBoard() {
  const features = [
    {
      icon: <Mic size={28} />,
      color: "bg-yellow-200",
      title: "Live Recording",
      desc: "Record directly in-browser. Real-time waveform + cross-talk detection built in.",
      rotate: -2,
    },
    {
      icon: <Users size={28} />,
      color: "bg-blue-200",
      title: "Speaker ID",
      desc: "Automatically diarize and identify each speaker from stored voice profiles.",
      rotate: 1.5,
    },
    {
      icon: <Wand2 size={28} />,
      color: "bg-pink-200",
      title: "AI Insights",
      desc: "Ask any question about your meeting. Get summaries, action items, and key points.",
      rotate: -1,
    },
    {
      icon: <FileText size={28} />,
      color: "bg-green-200",
      title: "Clean Transcripts",
      desc: "Word-level timestamps, speaker labels, and exportable formatted transcripts.",
      rotate: 2,
    },
    {
      icon: <Layers size={28} />,
      color: "bg-indigo-200",
      title: "Tab Audio Capture",
      desc: "Capture browser tab audio — perfect for recording remote calls and webinars.",
      rotate: -0.5,
    },
    {
      icon: <Zap size={28} />,
      color: "bg-red-200",
      title: "Instant Upload",
      desc: "Drop any audio or video file. We'll transcribe and diarize it automatically.",
      rotate: 1,
    },
  ];

  return (
    <section className="relative py-24 px-6 overflow-hidden">
      <GraphPaper />
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="font-display text-5xl md:text-6xl font-bold text-ink mb-3"
            style={{ filter: "url(#lp-squiggle-soft)" }}
          >
            The Blueprint.
          </motion.h2>
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="font-hand text-lg text-pencil"
          >
            Everything you need, scribbled into one clean dashboard.
          </motion.p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((f, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
            >
              <StickyNote color={f.color} rotate={f.rotate} className="h-full">
                <div
                  className="w-12 h-12 flex items-center justify-center border-2 border-ink mb-3 text-ink"
                  style={{
                    borderRadius: "10px 16px 12px 14px / 14px 10px 16px 12px",
                    background: "hsl(var(--card) / .5)",
                  }}
                >
                  {f.icon}
                </div>
                <h3 className="font-display text-xl font-bold text-ink mb-1">
                  {f.title}
                </h3>
                <p className="font-hand text-sm text-ink/70 leading-relaxed">
                  {f.desc}
                </p>
              </StickyNote>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── 8. Horizontal Showcase ────────────────────────────────────────────────
function SketchbookShowcase() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (!sectionRef.current || !triggerRef.current) return;

    const pin = gsap.to(sectionRef.current, {
      x: "-66%",
      ease: "none",
      scrollTrigger: {
        trigger: triggerRef.current,
        pin: true,
        scrub: 1,
        end: () => `+=${sectionRef.current!.offsetWidth}`,
      },
    });
    return () => {
      if (pin) pin.kill();
    };
  }, []);

  const slides = [
    {
      title: "Live Recording",
      tag: "#Record",
      color: "bg-blue-50",
      desc: "Real-time waveform, cross-talk alerts, and one-click submission.",
      icon: <Mic size={40} className="text-accent" />,
    },
    {
      title: "AI Chat Panel",
      tag: "#Insights",
      color: "bg-pink-50",
      desc: "Chat with your transcript. Ask questions, get summaries, extract action items.",
      icon: <MessageSquare size={40} className="text-accent" />,
    },
    {
      title: "Speaker Profiles",
      tag: "#Diarize",
      color: "bg-green-50",
      desc: "Train voice embeddings to automatically recognize who said what.",
      icon: <Users size={40} className="text-accent" />,
    },
  ];

  return (
    <div ref={triggerRef} className="overflow-hidden">
      <div
        ref={sectionRef}
        className="flex"
        style={{ width: `${slides.length * 100}vw` }}
      >
        {slides.map((slide, i) => (
          <div
            key={i}
            className={cn(
              "w-screen h-screen flex flex-col items-center justify-center px-12 relative",
              slide.color
            )}
          >
            <GraphPaper />
            <div className="max-w-lg text-center">
              <span className="badge badge-yellow mb-4 inline-block">
                Draft {i + 1} — {slide.tag}
              </span>
              <div className="mb-6">{slide.icon}</div>
              <h2
                className="font-display text-5xl font-bold text-ink mb-4"
                style={{ filter: "url(#lp-squiggle-soft)" }}
              >
                {slide.title}
              </h2>
              <p className="font-hand text-lg text-pencil mb-8">{slide.desc}</p>
              <Link to="/signup">
                <SketchButton variant="primary" className="gap-2">
                  Try it free <ArrowRight size={16} />
                </SketchButton>
              </Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── 9. Process Path ──────────────────────────────────────────────────────
function ProcessPath() {
  const container = useRef<HTMLDivElement>(null);
  const pathRef = useRef<SVGPathElement>(null);

  useLayoutEffect(() => {
    const path = pathRef.current;
    if (!path) return;
    const totalLength = path.getTotalLength();
    gsap.set(path, {
      strokeDasharray: totalLength,
      strokeDashoffset: totalLength,
    });
    gsap.to(path, {
      strokeDashoffset: 0,
      ease: "none",
      scrollTrigger: {
        trigger: container.current,
        start: "top center",
        end: "bottom center",
        scrub: 1,
      },
    });
  }, []);

  const steps = [
    { title: "Record or Upload", icon: <Mic size={36} />, desc: "Start a live recording or drop any audio/video file." },
    { title: "AI Transcribes", icon: <FileText size={36} />, desc: "Automatic transcription with word-level timestamps." },
    { title: "Speaker ID", icon: <Users size={36} />, desc: "Diarization engine maps every word to the right voice." },
    { title: "Get Insights", icon: <Sparkles size={36} />, desc: "Chat with AI, export summaries, and share results." },
  ];

  return (
    <section ref={container} className="relative py-24 px-6 overflow-hidden">
      <GraphPaper />
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="font-display text-5xl md:text-6xl font-bold text-ink mb-3"
            style={{ filter: "url(#lp-squiggle-soft)" }}
          >
            How It Works.
          </motion.h2>
        </div>

        <div className="relative">
          {/* Scroll-driven path */}
          <svg
            className="absolute left-1/2 -translate-x-1/2 top-0 h-full pointer-events-none"
            width="100"
            viewBox="0 0 60 800"
            preserveAspectRatio="none"
          >
            <path
              ref={pathRef}
              d="M30 0 Q 60 100, 30 200 Q 0 300, 30 400 Q 60 500, 30 600 Q 0 700, 30 800"
              stroke="hsl(var(--accent))"
              strokeWidth="3"
              fill="none"
              strokeLinecap="round"
            />
          </svg>

          <div className="flex flex-col gap-24">
            {steps.map((s, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: i % 2 === 0 ? -40 : 40 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ duration: 0.6 }}
                className={cn(
                  "flex items-start gap-8",
                  i % 2 !== 0 && "flex-row-reverse"
                )}
              >
                <div
                  className="w-20 h-20 flex-shrink-0 flex items-center justify-center bg-card border-2 border-ink shadow-sketch text-ink"
                  style={{
                    borderRadius: "14px 20px 16px 22px / 20px 14px 22px 16px",
                    filter: "url(#lp-squiggle-soft)",
                  }}
                >
                  {s.icon}
                </div>
                <div className="flex-1 max-w-sm">
                  <h3 className="font-display text-2xl font-bold text-ink mb-2">
                    {String(i + 1).padStart(2, "0")}. {s.title}
                  </h3>
                  <p className="font-hand text-base text-pencil leading-relaxed">
                    {s.desc}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

// ─── 10. Testimonials ─────────────────────────────────────────────────────
function ClientScribbles() {
  const testimonials = [
    {
      text: "Our team stopped taking notes the moment we started using VoiceSum. It just works.",
      author: "Sarah K., Product Lead",
      color: "bg-blue-100",
    },
    {
      text: "The speaker diarization is shockingly accurate. Finally know who said what.",
      author: "James T., Engineering Manager",
      color: "bg-yellow-100",
    },
    {
      text: "I upload every client call. The AI summary saves me 30 minutes of write-up per call.",
      author: "Priya M., Consultant",
      color: "bg-pink-100",
    },
    {
      text: "Cross-talk detection is my favourite hidden gem. Keeps meetings civil!",
      author: "Marco R., Facilitator",
      color: "bg-green-100",
    },
    {
      text: "The sketchy UI is so charming. Never thought I'd enjoy a transcription tool.",
      author: "Lily C., UX Designer",
      color: "bg-indigo-100",
    },
  ];

  return (
    <section className="relative py-24 px-6 overflow-hidden">
      <GraphPaper />
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-14">
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="font-blueprint text-sm uppercase tracking-widest text-pencil mb-2"
          >
            User Feedback
          </motion.p>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="font-display text-5xl font-bold text-ink"
            style={{ filter: "url(#lp-squiggle-soft)" }}
          >
            Scribbles from the Field.
          </motion.h2>
        </div>

        <div className="columns-1 sm:columns-2 lg:columns-3 gap-6 space-y-6">
          {testimonials.map((t, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.07 }}
              className="break-inside-avoid"
            >
              <div
                className={cn(
                  "p-6 border-2 border-ink shadow-sketch font-hand",
                  t.color
                )}
                style={{
                  borderRadius: "12px 20px 14px 18px / 18px 14px 20px 12px",
                  transform: `rotate(${(i % 2 === 0 ? 1 : -1) * (0.5 + (i % 3) * 0.3)}deg)`,
                }}
              >
                <Star size={14} className="text-accent mb-2" fill="currentColor" />
                <p className="text-ink text-base leading-relaxed mb-4">
                  "{t.text}"
                </p>
                <div className="flex items-center gap-2">
                  <div
                    className="w-6 h-6 bg-accent/20 border border-ink flex items-center justify-center text-xs font-bold text-accent"
                    style={{ borderRadius: "50%" }}
                  >
                    {t.author[0]}
                  </div>
                  <span className="text-sm text-pencil font-bold">
                    — {t.author}
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}



// ─── 12. Blueprint Footer ─────────────────────────────────────────────────
function BlueprintFooter() {
  return (
    <footer className="relative overflow-hidden border-t-2 border-ink">
      {/* Blueprint dark BG */}
      <div
        className="absolute inset-0 bg-blueprint"
        style={{
          backgroundImage: `
            linear-gradient(hsl(200 70% 70% / 0.08) 1px, transparent 1px),
            linear-gradient(90deg, hsl(200 70% 70% / 0.08) 1px, transparent 1px)
          `,
          backgroundSize: "30px 30px",
        }}
      />

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-20">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mb-12">
          {/* Brand */}
          <div>
            <h2
              className="font-display text-6xl font-bold text-blue-100 mb-4 leading-tight"
              style={{ filter: "url(#lp-squiggle-soft)" }}
            >
              Let's <br />
              <span className="text-accent">Record.</span>
            </h2>
            <p className="font-hand text-blue-200/80 max-w-xs leading-relaxed">
              Your next great insight starts as a voice memo. VoiceSum turns
              every conversation into a searchable, shareable record.
            </p>
          </div>

          {/* Links */}
          <div className="flex gap-16">
            <div>
              <p className="font-blueprint text-xs uppercase tracking-widest text-blue-300 mb-4">
                Navigation
              </p>
              <div className="space-y-2">
                {["Home", "Dashboard", "Upload", "History"].map((l) => (
                  <p key={l}>
                    <Link
                      to="/"
                      className="font-hand text-blue-100/70 hover:text-accent transition-colors"
                    >
                      {l}
                    </Link>
                  </p>
                ))}
              </div>
            </div>
            <div>
              <p className="font-blueprint text-xs uppercase tracking-widest text-blue-300 mb-4">
                Product
              </p>
              <div className="space-y-2">
                {["Sign Up", "Login", "Settings", "Add Voice"].map((l) => (
                  <p key={l}>
                    <Link
                      to="/signup"
                      className="font-hand text-blue-100/70 hover:text-accent transition-colors"
                    >
                      {l}
                    </Link>
                  </p>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Coordinate footer bar */}
        <div className="border-t border-blue-400/20 pt-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <p className="font-blueprint text-xs text-blue-300/60">
            © 2026 VoiceSum Inc. — All conversations transcribed with care.
          </p>
          <p className="font-blueprint text-xs text-blue-300/40">
            Coordinate System: 0.0000N / 0.0000W — Blueprint v2.0
          </p>
        </div>
      </div>
    </footer>
  );
}

// ─── 13. Navbar ───────────────────────────────────────────────────────────
function Navbar() {
  const [scrolled, setScrolled] = React.useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div
      style={{
        position: "sticky", top: 0, zIndex: 50,
        display: "flex", justifyContent: "center",
        padding: scrolled ? "10px 16px" : "14px 16px",
        background: scrolled ? "hsl(var(--background) / .7)" : "transparent",
        backdropFilter: scrolled ? "blur(20px)" : "none",
        WebkitBackdropFilter: scrolled ? "blur(20px)" : "none",
        borderBottom: scrolled ? "1px solid hsl(var(--ink) / .07)" : "none",
        transition: "all .35s cubic-bezier(0.4,0,0.2,1)",
      }}
    >
      <nav
        style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr auto",
          alignItems: "center",
          width: "100%", maxWidth: scrolled ? "860px" : "1080px",
          padding: scrolled ? "8px 16px 8px 12px" : "10px 20px 10px 14px",
          borderRadius: "999px",
          background: scrolled
            ? "hsl(var(--card) / .92)"
            : "hsl(var(--card) / .6)",
          border: "1.5px solid hsl(var(--ink) / .1)",
          boxShadow: scrolled
            ? "0 8px 32px hsl(var(--ink) / .1), 0 1px 4px hsl(var(--ink) / .06), inset 0 1px 0 hsl(255,100%,100% / .08)"
            : "0 2px 16px hsl(var(--ink) / .07), inset 0 1px 0 hsl(255,100%,100% / .06)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          transition: "all .35s cubic-bezier(0.4,0,0.2,1)",
        }}
      >
        {/* Logo */}
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 8, textDecoration: "none" }}>
          <div style={{ position: "relative" }}>
            <Zap
              size={22}
              fill="currentColor"
              className="text-accent animate-float"
            />
            <span style={{
              position: "absolute", top: -2, right: -2,
              width: 6, height: 6, borderRadius: "50%",
              background: "hsl(var(--accent))",
              boxShadow: "0 0 5px hsl(var(--accent))",
            }} className="animate-pulse-rec" />
          </div>
          <span className="font-display text-2xl font-bold text-ink">
            Voice<span className="text-accent">Sum</span>
          </span>
        </Link>

        {/* Center nav links — truly centered via 1fr column + mx-auto */}
        <div className="hidden md:flex items-center justify-center" style={{ gap: 2 }}>
          {[
            { label: "Features", href: "#features" },
            { label: "How It Works", href: "#process" }
          ].map((item) => (
            <a
              key={item.label}
              href={item.href}
              className="font-hand"
              style={{
                fontSize: "1rem", fontWeight: 600,
                color: "hsl(var(--pencil))",
                textDecoration: "none",
                padding: "6px 14px",
                borderRadius: "999px",
                transition: "all .18s",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.color = "hsl(var(--accent))";
                (e.currentTarget as HTMLElement).style.background = "hsl(var(--accent) / .09)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.color = "hsl(var(--pencil))";
                (e.currentTarget as HTMLElement).style.background = "transparent";
              }}
            >
              {item.label}
            </a>
          ))}
        </div>

        {/* CTAs */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Link to="/login" style={{ textDecoration: "none" }}>
            <button
              className="font-hand"
              style={{
                fontWeight: 700, fontSize: ".93rem",
                padding: "7px 18px", borderRadius: "999px",
                border: "1.5px solid hsl(var(--ink) / .14)",
                background: "transparent",
                color: "hsl(var(--ink))",
                cursor: "pointer",
                transition: "all .18s",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.background = "hsl(var(--ink) / .06)";
                (e.currentTarget as HTMLElement).style.borderColor = "hsl(var(--ink) / .25)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.background = "transparent";
                (e.currentTarget as HTMLElement).style.borderColor = "hsl(var(--ink) / .14)";
              }}
            >
              Sign In
            </button>
          </Link>
          <Link to="/signup" style={{ textDecoration: "none" }}>
            <button
              className="font-hand"
              style={{
                fontWeight: 700, fontSize: ".93rem",
                padding: "7px 20px", borderRadius: "999px",
                border: "none",
                background: "linear-gradient(135deg, hsl(var(--accent)), hsl(var(--accent) / .75))",
                color: "white",
                cursor: "pointer",
                boxShadow: "0 2px 14px hsl(var(--accent) / .38)",
                transition: "all .18s",
                display: "flex", alignItems: "center", gap: 5,
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
                (e.currentTarget as HTMLElement).style.boxShadow = "0 6px 22px hsl(var(--accent) / .48)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
                (e.currentTarget as HTMLElement).style.boxShadow = "0 2px 14px hsl(var(--accent) / .38)";
              }}
            >
              Start Free →
            </button>
          </Link>
        </div>
      </nav>
    </div>
  );
}

// ─── 14. Main Landing Page ────────────────────────────────────────────────
export default function Landing() {
  // Lenis smooth scroll (scoped to this page)
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: "vertical",
      gestureOrientation: "vertical",
      smoothWheel: true,
    });

    lenis.on("scroll", ScrollTrigger.update);
    const raf = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(raf);
    gsap.ticker.lagSmoothing(0);

    return () => {
      gsap.ticker.remove(raf);
      lenis.destroy();
      ScrollTrigger.getAll().forEach((t) => t.kill());
    };
  }, []);

  return (
    <div className="relative bg-background text-foreground overflow-x-hidden">
      <SquiggleFilter />

      <Navbar />

      <main>
        <Hero />
        <TapeMarquee />

        <div id="features">
          <FeatureBoard />
        </div>

        <SketchbookShowcase />

        <div id="process">
          <ProcessPath />
        </div>

        <ClientScribbles />


      </main>

      <BlueprintFooter />
    </div>
  );
}
