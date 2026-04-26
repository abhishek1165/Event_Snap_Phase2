import React from 'react';
import { Shield, Zap, Lock, Globe, Clock, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';

const features = [
  { icon: Zap, title: 'Lightning Fast', desc: 'AI-powered matching delivers results in under 3 seconds, no matter how many photos.', color: 'text-yellow-400', bg: 'bg-yellow-400/10' },
  { icon: Shield, title: 'Privacy First', desc: 'Only face embeddings are stored — we never keep raw biometric data.', color: 'text-emerald-400', bg: 'bg-emerald-400/10' },
  { icon: Lock, title: 'Event Scoped', desc: 'Searches are strictly isolated to each event — zero cross-event leakage.', color: 'text-blue-400', bg: 'bg-blue-400/10' },
  { icon: Globe, title: 'Batch Upload', desc: 'Upload hundreds of photos at once with smart background processing.', color: 'text-violet-400', bg: 'bg-violet-400/10' },
  { icon: Clock, title: 'Live Status', desc: 'Real-time photo processing updates so you always know when events are ready.', color: 'text-cyan-400', bg: 'bg-cyan-400/10' },
  { icon: TrendingUp, title: 'Analytics', desc: 'Insights on attendee searches, match rates, and event photo coverage.', color: 'text-pink-400', bg: 'bg-pink-400/10' }
];

export default function Features() {
  return (
    <section id="features" className="relative py-28 bg-slate-950">
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
            Powerful Features for Every Event
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Everything you need to organize and share event photos seamlessly
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -4 }}
              className="group p-6 rounded-2xl bg-slate-900/50 border border-white/5 hover:border-white/10 transition-all"
            >
              <div className={`w-12 h-12 rounded-xl ${feature.bg} flex items-center justify-center mb-4`}>
                <feature.icon className={`w-6 h-6 ${feature.color}`} />
              </div>
              <h3 className="text-white font-semibold text-lg mb-2">{feature.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{feature.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
