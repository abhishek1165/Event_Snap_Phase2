import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Camera, Search, ArrowRight, Shield, Zap, Image } from 'lucide-react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import api from '@/utils/api';

export default function AttendeeEntry() {
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleFind = async (e) => {
    e.preventDefault();
    if (!code.trim()) return;
    setLoading(true);
    try {
      const response = await api.get(`/events/code/${code.toUpperCase().trim()}`);
      const event = response.data;
      
      if (event.status === 'processing') {
        toast.error('Event is still processing photos. Please try again later.');
        setLoading(false);
        return;
      }
      
      if (event.status === 'active' && event.total_photos === 0) {
        toast.error('This event has no photos yet. Please contact the organizer.');
        setLoading(false);
        return;
      }
      
      if (event.faces_detected === 0) {
        toast.warning('Note: No faces detected in photos yet. You can still try searching.');
      }
      
      toast.success(`Found event: ${event.title}`);
      navigate(`/attend/${event.id}/selfie`, { state: { event } });
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Event not found');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex">
      {/* Left brand panel */}
      <div className="hidden lg:flex lg:w-[45%] relative overflow-hidden flex-col justify-center px-16">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-600 via-violet-700 to-purple-800" />
        <div className="absolute inset-0 opacity-[0.07]" style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-72 h-72 bg-white/5 rounded-full blur-3xl" />
        <div className="relative z-10">
          <Link to="/Landing" className="flex items-center gap-2.5 mb-14">
            <div className="w-9 h-9 rounded-xl bg-white/20 flex items-center justify-center">
              <Camera className="w-5 h-5 text-white" />
            </div>
            <span className="text-white font-bold text-xl">FaceShot</span>
          </Link>
          <h2 className="text-4xl font-bold text-white mb-4 leading-tight">Find every photo<br />of yourself</h2>
          <p className="text-indigo-200 text-lg mb-12">Enter your event code and take one selfie. AI instantly finds all your photos.</p>
          <div className="space-y-6">
            {[
              { icon: Search, title: 'Enter Event Code', desc: 'Get the code from your event organizer' },
              { icon: Camera, title: 'Take a Selfie', desc: 'Quick photo for face recognition' },
              { icon: Image, title: 'Get Your Photos', desc: 'Instantly see all photos with your face' },
            ].map(step => (
              <div key={step.title} className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-xl bg-white/15 flex items-center justify-center flex-shrink-0">
                  <step.icon className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-white font-medium">{step.title}</p>
                  <p className="text-indigo-200/80 text-sm">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="flex gap-6 mt-12 pt-8 border-t border-white/15">
            {[{ icon: Shield, label: 'Privacy First' }, { icon: Zap, label: 'Under 3 seconds' }].map(({ icon: Ic, label }) => (
              <div key={label} className="flex items-center gap-2 text-indigo-200/80 text-sm">
                {React.createElement(Ic, { className: "w-4 h-4" })}{label}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right form panel */}
      <div className="flex-1 flex items-center justify-center p-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md">
          <Link to="/Landing" className="flex items-center gap-2.5 mb-12 lg:hidden">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <Camera className="w-5 h-5 text-white" />
            </div>
            <span className="text-white font-bold text-xl">FaceShot</span>
          </Link>

          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Find Your Photos</h1>
            <p className="text-slate-500">Enter the event code from your organizer</p>
          </div>

          <form onSubmit={handleFind} className="space-y-4">
            <div>
              <label className="text-slate-400 text-sm mb-2 block">Event Code</label>
              <input value={code} onChange={e => setCode(e.target.value.toUpperCase())}
                placeholder="e.g., ABC123"
                maxLength={8}
                className="w-full bg-white/5 border border-white/8 rounded-xl px-4 py-4 text-white placeholder-slate-700 focus:outline-none focus:border-indigo-500 text-xl font-mono tracking-[0.3em] uppercase text-center transition-colors" />
            </div>
            <button type="submit" disabled={loading || !code.trim()}
              className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold py-4 rounded-xl transition-all disabled:opacity-40 shadow-xl shadow-indigo-500/20">
              {loading ? (
                <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Searching...</>
              ) : (<>Find My Photos <ArrowRight className="w-4 h-4" /></>)}
            </button>
          </form>

          <p className="text-center text-slate-600 text-sm mt-8">
            Are you an organizer?{' '}
            <Link to="/OrganizerDashboard" className="text-indigo-400 hover:text-indigo-300 transition-colors">Go to Dashboard →</Link>
          </p>
        </motion.div>
      </div>
    </div>
  );
}
