import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Camera, Plus, LogOut, Calendar, Image as ImageIcon,
  LayoutGrid, Zap, Hash, ArrowRight, X, ChevronRight
} from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { toast } from 'sonner';
import api from '@/utils/api';
import { cn } from '@/lib/utils';

/* ─── Stagger animation variants ─── */
const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.06 } }
};
const cardVariants = {
  hidden: { opacity: 0, scale: 0.92, y: 16 },
  visible: { opacity: 1, scale: 1, y: 0, transition: { ease: [0.34, 1.56, 0.64, 1], duration: 0.4 } }
};

/* ─── Status badge config ─── */
const STATUS_CONFIG = {
  completed:  { label: 'Completed',  dot: 'bg-green-500',  text: 'text-green-400',  bg: 'bg-green-500/10' },
  processing: { label: 'Processing', dot: 'bg-amber-400',  text: 'text-amber-400',  bg: 'bg-amber-400/10' },
  active:     { label: 'Active',     dot: 'bg-blue-400',   text: 'text-blue-400',   bg: 'bg-blue-400/10'  },
};

const StatusBadge = ({ status }) => {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.active;
  return (
    <span className={cn('inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold tracking-wide', cfg.bg, cfg.text)}>
      <span className={cn('w-1.5 h-1.5 rounded-full', cfg.dot)} />
      {cfg.label}
    </span>
  );
};

/* ─── Stat Card ─── */
const StatCard = ({ icon: Icon, label, value, accent }) => (
  <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
    <div className={cn('absolute inset-0 opacity-[0.04]', accent)} />
    <div className="relative flex items-center gap-4">
      <div className={cn('flex h-10 w-10 items-center justify-center rounded-lg', accent, 'bg-opacity-15')}>
        <Icon className={cn('h-5 w-5', accent.replace('bg-', 'text-'))} strokeWidth={1.8} />
      </div>
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">{label}</p>
        <p className="mt-0.5 text-2xl font-bold text-white tabular-nums">{value}</p>
      </div>
    </div>
  </div>
);

const OrganizerDashboard = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newEvent, setNewEvent] = useState({ title: '', description: '', date: '' });

  useEffect(() => {
    const userData = JSON.parse(localStorage.getItem('user') || '{}');
    setUser(userData);
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      const response = await api.get('/events');
      setEvents(response.data);
    } catch (error) {
      toast.error('Failed to load events');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateEvent = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post('/events', newEvent);
      toast.success('Event created.');
      setEvents([response.data, ...events]);
      setCreateDialogOpen(false);
      setNewEvent({ title: '', description: '', date: '' });
    } catch (error) {
      toast.error('Failed to create event.');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/');
  };

  /* ─── Loading skeleton ─── */
  if (loading) {
    return (
      <div className="min-h-screen bg-[#020617] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="h-10 w-10 rounded-full border-2 border-transparent border-t-green-500 animate-spin" />
          <p className="text-slate-500 text-sm font-medium tracking-wide">Loading dashboard…</p>
        </div>
      </div>
    );
  }

  const totalPhotos = events.reduce((acc, e) => acc + (e.total_photos || 0), 0);
  const activeEvents = events.filter(e => e.status !== 'completed').length;

  return (
    <div className="min-h-screen bg-[#020617] text-slate-50" style={{ fontFamily: 'Outfit, sans-serif' }}>

      {/* ── Ambient glow blobs ── */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 h-[500px] w-[500px] rounded-full bg-green-500/[0.05] blur-3xl" />
        <div className="absolute top-1/3 right-0 h-[400px] w-[400px] rounded-full bg-indigo-500/[0.07] blur-3xl" />
      </div>

      {/* ── Top Navigation Bar ── */}
      <header className="relative z-10 border-b border-slate-800 bg-slate-950/80 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
          {/* Brand */}
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-green-500 shadow-[0_0_16px_rgba(34,197,94,0.4)]">
              <Camera className="h-4 w-4 text-[#020617]" strokeWidth={2.5} />
            </div>
            <span className="font-bold text-base tracking-tight">FaceShot</span>
            <span className="ml-1 rounded-full bg-slate-800 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
              Pro
            </span>
          </div>

          {/* Right side */}
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5">
              <div className="h-2 w-2 rounded-full bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.8)] animate-pulse" />
              <span className="text-xs font-medium text-slate-400">{user?.name || 'Organizer'}</span>
            </div>
            <button
              data-testid="logout-button"
              onClick={handleLogout}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-slate-500 hover:bg-slate-800 hover:text-slate-200 transition-colors"
            >
              <LogOut className="h-4 w-4" />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="relative z-10 mx-auto max-w-7xl px-6 py-10">

        {/* Page header */}
        <div className="mb-10 flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-2">Dashboard</p>
            <h1 className="text-4xl font-bold tracking-tight text-white">My Events</h1>
            <p className="mt-1.5 text-slate-400">Manage your events, photos, and access codes.</p>
          </div>

          {/* Create Event Dialog */}
          <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
            <DialogTrigger asChild>
              <button
                data-testid="create-event-button"
                className="flex items-center gap-2 rounded-xl bg-green-500 px-5 py-3 text-sm font-bold text-[#020617] shadow-[0_0_24px_rgba(34,197,94,0.25)] hover:bg-green-400 hover:shadow-[0_0_32px_rgba(34,197,94,0.4)] transition-all duration-200"
              >
                <Plus className="h-4 w-4" strokeWidth={2.5} />
                New Event
              </button>
            </DialogTrigger>
            <DialogContent className="border border-slate-800 bg-slate-950 text-white sm:max-w-md">
              <DialogHeader>
                <DialogTitle className="text-xl font-bold tracking-tight">Create New Event</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleCreateEvent} className="mt-4 space-y-5">
                {[
                  { id: 'event-title', testId: 'event-title-input', label: 'Event Title', type: 'text', placeholder: "John & Jane's Wedding", key: 'title', required: true },
                  { id: 'event-description', testId: 'event-description-input', label: 'Description', type: 'text', placeholder: 'Grand Hotel Ballroom celebration…', key: 'description' },
                  { id: 'event-date', testId: 'event-date-input', label: 'Event Date', type: 'date', placeholder: '', key: 'date' },
                ].map(({ id, testId, label, type, placeholder, key, required }) => (
                  <div key={id} className="space-y-1.5">
                    <label htmlFor={id} className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                      {label}
                    </label>
                    <input
                      id={id}
                      data-testid={testId}
                      type={type}
                      placeholder={placeholder}
                      value={newEvent[key]}
                      onChange={(e) => setNewEvent({ ...newEvent, [key]: e.target.value })}
                      required={required}
                      className="w-full rounded-lg border border-slate-800 bg-slate-900 px-4 py-3 text-sm text-white placeholder:text-slate-600 focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/30 transition-all"
                    />
                  </div>
                ))}
                <button
                  data-testid="create-event-submit-button"
                  type="submit"
                  className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg bg-green-500 py-3 text-sm font-bold text-[#020617] hover:bg-green-400 transition-colors"
                >
                  Create Event
                  <ArrowRight className="h-4 w-4" />
                </button>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        {/* ── Stats Row ── */}
        <div className="mb-8 grid grid-cols-2 gap-4 sm:grid-cols-3">
          <StatCard icon={LayoutGrid} label="Total Events" value={events.length} accent="bg-indigo-500" />
          <StatCard icon={Zap} label="Active"       value={activeEvents}    accent="bg-green-500" />
          <StatCard icon={ImageIcon} label="Total Photos" value={totalPhotos}    accent="bg-amber-400" />
        </div>

        {/* ── Events Grid ── */}
        {events.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-slate-800 bg-slate-900/30 py-24 text-center"
          >
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900">
              <Calendar className="h-8 w-8 text-slate-600" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">No events yet</h3>
            <p className="text-slate-500 mb-6 max-w-xs">Create your first event to start collecting and sharing photos with attendees.</p>
            <button
              data-testid="empty-create-event-button"
              onClick={() => setCreateDialogOpen(true)}
              className="flex items-center gap-2 rounded-xl bg-green-500 px-5 py-2.5 text-sm font-bold text-[#020617] hover:bg-green-400 transition-colors shadow-[0_0_20px_rgba(34,197,94,0.2)]"
            >
              <Plus className="h-4 w-4" strokeWidth={2.5} />
              Create your first event
            </button>
          </motion.div>
        ) : (
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
          >
            {events.map((event) => (
              <motion.div
                key={event.id}
                variants={cardVariants}
                data-testid={`event-card-${event.id}`}
                onClick={() => navigate(`/events/${event.id}`)}
                className="group relative flex cursor-pointer flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur-sm transition-all duration-300 hover:border-slate-700 hover:bg-slate-900 hover:-translate-y-0.5 hover:shadow-[0_8px_32px_rgba(0,0,0,0.4)]"
              >
                {/* Top: icon + status */}
                <div className="mb-5 flex items-start justify-between">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-indigo-500/10 border border-indigo-500/20">
                    <ImageIcon className="h-5 w-5 text-indigo-400" strokeWidth={1.8} />
                  </div>
                  <StatusBadge status={event.status || 'active'} />
                </div>

                {/* Title & Description */}
                <h3 className="text-base font-bold text-white mb-1 line-clamp-1">{event.title}</h3>
                {event.description && (
                  <p className="text-sm text-slate-500 line-clamp-2 mb-4">{event.description}</p>
                )}
                {!event.description && <div className="mb-4" />}

                {/* Footer */}
                <div className="mt-auto flex items-center justify-between border-t border-slate-800 pt-4">
                  <div className="flex items-center gap-1.5">
                    <Hash className="h-3.5 w-3.5 text-slate-600" />
                    <span className="font-mono text-sm font-semibold text-green-400 tracking-wider">
                      {event.event_code}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 text-slate-500 text-xs">
                    <ImageIcon className="h-3.5 w-3.5" />
                    <span>{event.total_photos ?? 0} photos</span>
                  </div>
                </div>

                {/* Hover chevron arrow */}
                <ChevronRight className="absolute right-5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-700 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all duration-200" />
              </motion.div>
            ))}
          </motion.div>
        )}
      </main>
    </div>
  );
};

export default OrganizerDashboard;
