import React from 'react';
import { Link } from 'react-router-dom';
import { Camera, Sparkles, ArrowRight, Users, Zap, Shield } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Hero() {
  return (
    <section className="relative min-h-screen bg-slate-950 flex flex-col overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-950/60 via-slate-950 to-violet-950/40" />
        <div className="absolute top-0 left-1/3 w-[500px] h-[500px] bg-indigo-600/8 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-violet-600/8 rounded-full blur-3xl" />
        <div className="absolute inset-0 opacity-[0.03]"
          style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)', backgroundSize: '64px 64px' }}
        />
      </div>

      {/* Navbar */}
      <nav className="relative z-10 max-w-7xl mx-auto w-full px-6 md:px-10 py-5 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <Camera className="w-5 h-5 text-white" />
          </div>
          <span className="text-white font-bold text-xl">FaceShot</span>
        </div>
        <div className="hidden md:flex items-center gap-8">
          <a href="#how-it-works" className="text-slate-400 hover:text-white text-sm transition-colors">How it works</a>
          <a href="#features" className="text-slate-400 hover:text-white text-sm transition-colors">Features</a>
          <Link to="/attendjoin" className="text-slate-400 hover:text-white text-sm transition-colors">Find My Photos</Link>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/auth" className="text-slate-300 hover:text-white text-sm px-4 py-2 rounded-lg hover:bg-white/5 transition-all">Sign In</Link>
          <Link to="/auth" className="text-sm font-semibold bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white px-5 py-2.5 rounded-xl transition-all shadow-lg shadow-indigo-500/25">
            Get Started
          </Link>
        </div>
      </nav>

      {/* Hero Content */}
      <div className="relative z-10 flex-1 flex items-center">
        <div className="max-w-7xl mx-auto px-6 md:px-10 w-full py-16">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            {/* Left */}
            <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
              <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/20 rounded-full px-4 py-1.5 mb-7">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                <span className="text-indigo-300 text-xs font-medium tracking-wide">AI-Powered Face Recognition</span>
              </div>
              <h1 className="text-5xl md:text-6xl font-bold text-white leading-[1.08] mb-6 tracking-tight">
                Find Your Face in
                <span className="block bg-gradient-to-r from-indigo-400 via-violet-400 to-cyan-400 bg-clip-text text-transparent mt-1">
                  Every Event Photo
                </span>
              </h1>
              <p className="text-slate-400 text-lg leading-relaxed mb-8 max-w-lg">
                Stop searching through thousands of photos manually. One selfie is all it takes — our AI instantly finds every photo of you.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link to="/auth" className="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold px-6 py-3.5 rounded-xl transition-all shadow-xl shadow-indigo-500/25 text-sm">
                  For Organizers <ArrowRight className="w-4 h-4" />
                </Link>
                <Link to="/attendjoin" className="inline-flex items-center gap-2 border border-white/10 hover:border-white/20 text-white font-medium px-6 py-3.5 rounded-xl transition-all hover:bg-white/5 text-sm">
                  Find My Photos
                </Link>
              </div>
              <div className="flex flex-wrap gap-6 mt-10 pt-8 border-t border-white/5">
                <div className="flex items-center gap-2 text-slate-400 text-sm"><Users className="w-4 h-4 text-indigo-400" />50K+ Photos Processed</div>
                <div className="flex items-center gap-2 text-slate-400 text-sm"><Zap className="w-4 h-4 text-indigo-400" />Results in 3 seconds</div>
                <div className="flex items-center gap-2 text-slate-400 text-sm"><Shield className="w-4 h-4 text-indigo-400" />Privacy-first design</div>
              </div>
            </motion.div>

            {/* Right - Mockup */}
            <motion.div initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8, delay: 0.2 }} className="relative hidden md:block">
              <div className="relative">
                <div className="bg-slate-900/80 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
                  <img src="https://images.unsplash.com/photo-1540575467063-178a50c2df87?w=600&h=380&fit=crop" alt="Event" className="w-full h-60 object-cover opacity-75" />
                  <div className="p-5 flex items-center justify-between">
                    <div>
                      <p className="text-slate-500 text-xs mb-1">Tech Conference 2026</p>
                      <p className="text-white font-bold text-lg">312 Photos Found</p>
                    </div>
                    <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs px-3 py-1.5 rounded-full font-semibold">✓ 18 Matched</div>
                  </div>
                </div>
                <motion.div animate={{ y: [0, -7, 0] }} transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                  className="absolute -top-5 -right-5 bg-slate-900 border border-indigo-500/30 rounded-2xl px-4 py-3 shadow-2xl shadow-black/50">
                  <p className="text-indigo-400 text-xs font-medium mb-0.5">Face detected</p>
                  <p className="text-white text-sm font-bold">98.7% confidence</p>
                </motion.div>
                <motion.div animate={{ y: [0, 6, 0] }} transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                  className="absolute -bottom-5 -left-5 bg-slate-900 border border-violet-500/30 rounded-2xl px-4 py-3 shadow-2xl shadow-black/50 flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-violet-500/15 flex items-center justify-center">
                    <Camera className="w-5 h-5 text-violet-400" />
                  </div>
                  <div>
                    <p className="text-slate-500 text-xs">Selfie uploaded</p>
                    <p className="text-white text-sm font-semibold">Matching...</p>
                  </div>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
