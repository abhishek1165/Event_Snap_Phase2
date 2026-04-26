import React from 'react';
import { motion } from 'framer-motion';

export default function HowItWorks() {
  const steps = [
    {
      number: '01',
      title: 'Upload Event Photos',
      desc: 'Organizers batch upload hundreds of photos with intelligent background processing',
      gradient: 'from-indigo-500 to-violet-600'
    },
    {
      number: '02',
      title: 'Attendee Takes Selfie',
      desc: 'Guests simply take a quick selfie for AI-powered face recognition',
      gradient: 'from-violet-500 to-purple-600'
    },
    {
      number: '03',
      title: 'Instant Matching',
      desc: 'AI instantly finds all photos containing each attendee\'s face in seconds',
      gradient: 'from-purple-500 to-pink-600'
    }
  ];

  return (
    <section id="how-it-works" className="relative py-28 bg-slate-950">
      <div className="absolute inset-0 bg-gradient-to-b from-slate-950 via-indigo-950/10 to-slate-950" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-6 md:px-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
            How It Works
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Three simple steps to transform your event photo experience
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.15 }}
              className="relative group"
            >
              <div className="bg-slate-900/50 backdrop-blur-xl border border-white/5 rounded-3xl p-8 hover:border-white/10 transition-all">
                <div className={`inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br ${step.gradient} mb-6 shadow-lg`}>
                  <span className="text-white font-bold text-lg">{step.number}</span>
                </div>
                <h3 className="text-white font-semibold text-2xl mb-3">{step.title}</h3>
                <p className="text-slate-400 leading-relaxed">{step.desc}</p>
              </div>
              
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-4 w-8 h-px bg-gradient-to-r from-white/20 to-transparent" />
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
