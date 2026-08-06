import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Plus, LogOut, Calendar, Image as ImageIcon, LayoutGrid, Zap, Hash, ArrowRight, QrCode, Copy, Tv, Check } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "sonner";
import api from "@/utils/api";
import { cn } from "@/lib/utils";
import { AppShell, AppHeader } from "@/components/AppShell";
import { Button, Card, Input, Label } from "@/components/brand/atoms";
import { STAGGER } from "@/design/tokens";
import QrCardGeneratorModal from "@/components/organizer/QrCardGeneratorModal";

/* ── Stagger (Framer Motion) ── */
const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: STAGGER } },
};
const cardVariants = {
  hidden: { opacity: 0, scale: 0.96, y: 16 },
  visible: { opacity: 1, scale: 1, y: 0, transition: { type: "spring", stiffness: 320, damping: 30 } },
};

/* ── Status badge config ── */
const STATUS_CONFIG = {
  completed: { label: "Ready", dot: "bg-match", text: "text-match", bg: "bg-match/10" },
  processing: { label: "Processing", dot: "bg-memory", text: "text-memory", bg: "bg-memory/10" },
  active: { label: "Active", dot: "bg-iris-2", text: "text-iris-2", bg: "bg-iris/10" },
  failed: { label: "Failed", dot: "bg-red-400", text: "text-red-400", bg: "bg-red-400/10" },
};

const StatusBadge = ({ status }) => {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.active;
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.7rem] font-semibold tracking-wide", cfg.bg, cfg.text)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", cfg.dot)} />
      {cfg.label}
    </span>
  );
};

const StatCard = ({ icon: Icon, label, value, accent }) => (
  <Card className="overflow-hidden p-5">
    <div className="flex items-center gap-4">
      <div className={cn("flex h-10 w-10 items-center justify-center rounded-lg", accent, "bg-opacity-15")}>
        <Icon className={cn("h-5 w-5", accent.replace("bg-", "text-"))} strokeWidth={1.75} />
      </div>
      <div>
        <p className="text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-text-faint">{label}</p>
        <p className="mt-0.5 text-2xl font-bold text-text tnum">{value}</p>
      </div>
    </div>
  </Card>
);

const OrganizerDashboard = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newEvent, setNewEvent] = useState({ title: "", description: "", date: "" });
  
  // QR Card Modal State
  const [qrModalOpen, setQrModalOpen] = useState(false);
  const [selectedQrEvent, setSelectedQrEvent] = useState(null);

  useEffect(() => {
    const userData = JSON.parse(localStorage.getItem("user") || "{}");
    setUser(userData);
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      const response = await api.get("/events");
      setEvents(response.data);
    } catch (error) {
      toast.error("Failed to load events");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateEvent = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post("/events", newEvent);
      toast.success("Event created successfully!");
      setEvents([response.data, ...events]);
      setCreateDialogOpen(false);
      setNewEvent({ title: "", description: "", date: "" });
    } catch (error) {
      toast.error("Failed to create event.");
    }
  };

  const handleCopyJoinLink = (e, eventCode) => {
    e.stopPropagation();
    const joinUrl = `${window.location.origin}/attendjoin?code=${eventCode}`;
    navigator.clipboard.writeText(joinUrl);
    toast.success("Attendee Join Link copied to clipboard!");
  };

  const handleOpenQrModal = (e, eventObj) => {
    e.stopPropagation();
    setSelectedQrEvent(eventObj);
    setQrModalOpen(true);
  };

  const handleOpenSlideshow = (e, eventId) => {
    e.stopPropagation();
    navigate(`/events/${eventId}/slideshow`);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/");
  };

  if (loading) {
    return (
      <AppShell>
        <div className="flex min-h-[60dvh] items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <div className="h-10 w-10 rounded-full border-2 border-frame border-t-iris-2 animate-spin" />
            <p className="text-sm font-medium tracking-wide text-text-faint">Loading dashboard…</p>
          </div>
        </div>
      </AppShell>
    );
  }

  const totalPhotos = events.reduce((acc, e) => acc + (e.total_photos || 0), 0);
  const activeEvents = events.filter((e) => e.status !== "completed").length;

  return (
    <AppShell
      header={
        <AppHeader
          onLogoClick={() => navigate("/dashboard")}
          right={
            <>
              <div className="hidden items-center gap-2 rounded-full border border-frame-soft bg-surface-2 px-3 py-1.5 sm:flex">
                <div className="h-2 w-2 rounded-full bg-match shadow-[0_0_6px_rgba(52,211,153,.7)]" />
                <span className="text-xs font-medium text-text-dim">{user?.name || "Organizer"}</span>
              </div>
              <button
                data-testid="logout-button"
                onClick={handleLogout}
                className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-text-faint transition-colors hover:bg-surface-2 hover:text-text"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Logout</span>
              </button>
            </>
          }
        />
      }
    >
      <div className="mx-auto max-w-7xl px-5 py-10 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-10 flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="mb-2 text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-text-faint">Dashboard</p>
            <h1 className="text-4xl font-semibold tracking-tight text-text">My events</h1>
            <p className="mt-1.5 text-text-dim">Manage events, download QR cards, and track guest photos.</p>
          </div>

          <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button data-testid="create-event-button" size="lg" className="gap-2">
                <Plus className="h-4 w-4" strokeWidth={2.5} /> New event
              </Button>
            </DialogTrigger>
            <DialogContent className="border-frame bg-surface text-text sm:max-w-md">
              <DialogHeader>
                <DialogTitle className="text-xl font-semibold tracking-tight">Create new event</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleCreateEvent} className="mt-4 space-y-5">
                {[
                  { id: "event-title", testId: "event-title-input", label: "Event title", type: "text", placeholder: "Jordan and Avery wedding", key: "title", required: true },
                  { id: "event-description", testId: "event-description-input", label: "Description", type: "text", placeholder: "Grand Hotel ballroom celebration", key: "description" },
                  { id: "event-date", testId: "event-date-input", label: "Event date", type: "date", placeholder: "", key: "date" },
                ].map(({ id, testId, label, type, placeholder, key, required }) => (
                  <div key={id} className="space-y-2">
                    <Label htmlFor={id}>{label}</Label>
                    <Input
                      id={id}
                      data-testid={testId}
                      type={type}
                      placeholder={placeholder}
                      value={newEvent[key]}
                      onChange={(e) => setNewEvent({ ...newEvent, [key]: e.target.value })}
                      required={required}
                      className="border-frame"
                    />
                  </div>
                ))}
                <Button type="submit" data-testid="create-event-submit-button" className="mt-2 w-full">
                  Create event <ArrowRight className="h-4 w-4" />
                </Button>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        {/* Stats */}
        <div className="mb-8 grid grid-cols-2 gap-4 sm:grid-cols-3">
          <StatCard icon={LayoutGrid} label="Total events" value={events.length} accent="bg-iris" />
          <StatCard icon={Zap} label="Active" value={activeEvents} accent="bg-match" />
          <StatCard icon={ImageIcon} label="Total photos" value={totalPhotos} accent="bg-memory" />
        </div>

        {/* Grid */}
        {events.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-frame py-24 text-center"
          >
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-frame bg-surface-2">
              <Calendar className="h-8 w-8 text-text-faint" />
            </div>
            <h3 className="mb-2 text-xl font-semibold text-text">No events yet</h3>
            <p className="mb-6 max-w-xs text-text-faint">Create your first event to start collecting and sharing photos with attendees.</p>
            <Button data-testid="empty-create-event-button" onClick={() => setCreateDialogOpen(true)} className="gap-2">
              <Plus className="h-4 w-4" strokeWidth={2.5} /> Create your first event
            </Button>
          </motion.div>
        ) : (
          <motion.div variants={containerVariants} initial="hidden" animate="visible" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {events.map((event) => (
              <motion.div
                key={event.id}
                variants={cardVariants}
                data-testid={`event-card-${event.id}`}
                onClick={() => navigate(`/events/${event.id}`)}
                className="group relative flex cursor-pointer flex-col overflow-hidden rounded-2xl border border-frame-soft bg-surface p-6 transition-all duration-300 hover:-translate-y-0.5 hover:border-frame"
              >
                <div className="mb-5 flex items-start justify-between">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-iris/20 bg-iris/10">
                    <ImageIcon className="h-5 w-5 text-iris-2" strokeWidth={1.75} />
                  </div>
                  <StatusBadge status={event.status || "active"} />
                </div>

                <h3 className="mb-1 line-clamp-1 text-base font-bold text-text">{event.title}</h3>
                {event.description ? (
                  <p className="mb-4 line-clamp-2 text-sm text-text-faint">{event.description}</p>
                ) : (
                  <div className="mb-4" />
                )}

                {/* Quick Action Toolbar */}
                <div className="mb-4 flex items-center gap-1.5 rounded-xl border border-frame-soft bg-surface-2/60 p-1.5">
                  <button
                    title="Print / Download QR Table Card"
                    onClick={(e) => handleOpenQrModal(e, event)}
                    className="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-semibold text-text-dim hover:bg-surface-3 hover:text-text transition-colors"
                  >
                    <QrCode className="h-3.5 w-3.5 text-iris-2" />
                    <span>QR Cards</span>
                  </button>

                  <button
                    title="Copy Join Link"
                    onClick={(e) => handleCopyJoinLink(e, event.event_code)}
                    className="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-semibold text-text-dim hover:bg-surface-3 hover:text-text transition-colors"
                  >
                    <Copy className="h-3.5 w-3.5" />
                    <span>Link</span>
                  </button>

                  <button
                    title="Launch Live Slideshow"
                    onClick={(e) => handleOpenSlideshow(e, event.id)}
                    className="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-semibold text-text-dim hover:bg-surface-3 hover:text-text transition-colors"
                  >
                    <Tv className="h-3.5 w-3.5 text-match" />
                    <span>Slideshow</span>
                  </button>
                </div>

                <div className="mt-auto flex items-center justify-between border-t border-frame-soft pt-4">
                  <div className="flex items-center gap-1.5">
                    <Hash className="h-3.5 w-3.5 text-text-faint" />
                    <span className="font-mono text-sm font-semibold tracking-wider text-match">{event.event_code}</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs text-text-faint">
                    <ImageIcon className="h-3.5 w-3.5" />
                    <span className="tnum">{event.total_photos ?? 0} photos</span>
                  </div>
                </div>

                <ArrowRight className="absolute right-5 top-6 h-4 w-4 text-text-faint opacity-0 transition-all duration-200 group-hover:translate-x-0 group-hover:opacity-100" />
              </motion.div>
            ))}
          </motion.div>
        )}

        {/* QR Card Generator Modal */}
        <QrCardGeneratorModal
          open={qrModalOpen}
          onOpenChange={setQrModalOpen}
          event={selectedQrEvent}
        />
      </div>
    </AppShell>
  );
};

export default OrganizerDashboard;
