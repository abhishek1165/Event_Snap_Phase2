import React from 'react';
import { motion } from 'framer-motion';
import Hero from '../components/landing/Hero';
import HowItWorks from '../components/landing/HowItWorks';
import Features from '../components/landing/Features';
import LandingFooter from '../components/landing/LandingFooter';

export default function Landing() {
  return (
    <div className="bg-slate-950">
      <Hero />
      <HowItWorks />
      <Features />

      {/* CTA Section */}
      <section className="relative py-28 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 via-violet-600 to-purple-700" />
        <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)', backgroundSize: '48px 48px' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-white/5 rounded-full blur-3xl" />
        <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
              Ready to transform your event photos?
            </h2>
            <p className="text-indigo-200 text-lg mb-10 max-w-xl mx-auto">
              Join thousands of event organizers using AI-powered face recognition
            </p>
            <div className="flex flex-wrap gap-4 justify-center">
              <a href="/auth" className="bg-white text-indigo-700 font-bold px-8 py-4 rounded-xl hover:bg-indigo-50 transition-colors shadow-2xl text-sm">
                Get Started Free →
              </a>
              <a href="/attendjoin" className="border border-white/30 text-white font-medium px-8 py-4 rounded-xl hover:bg-white/10 transition-colors text-sm">
                Find My Photos
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      <LandingFooter />
    </div>
  );
}
