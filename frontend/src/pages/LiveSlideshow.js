import React, { useState, useEffect, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { QRCodeSVG } from "qrcode.react";
import { ArrowLeft, Play, Pause, Maximize, Minimize, QrCode } from "lucide-react";
import api, { getBackendUrl } from "@/utils/api";

export default function LiveSlideshow() {
  const { eventId } = useParams();
  const navigate = useNavigate();

  const [event, setEvent] = useState(null);
  const [photos, setPhotos] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [loading, setLoading] = useState(true);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

  const loadData = useCallback(async () => {
    try {
      const [eventRes, photosRes] = await Promise.all([
        api.get(`/events/${eventId}`),
        api.get(`/events/${eventId}/photos`),
      ]);
      setEvent(eventRes.data);
      setPhotos(photosRes.data);
    } catch (err) {
      console.error("Failed to load slideshow data", err);
    } finally {
      setLoading(false);
    }
  }, [eventId]);

  useEffect(() => {
    loadData();
    const pollInterval = setInterval(loadData, 10000); // refresh new photos every 10s
    return () => clearInterval(pollInterval);
  }, [loadData]);

  // Slideshow rotation timer
  useEffect(() => {
    if (!isPlaying || photos.length === 0) return;
    const timer = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % photos.length);
    }, 5000);
    return () => clearInterval(timer);
  }, [isPlaying, photos.length]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === " ") {
        e.preventDefault();
        setIsPlaying((prev) => !prev);
      } else if (e.key === "f" || e.key === "F") {
        toggleFullscreen();
      } else if (e.key === "Escape") {
        if (isFullscreen) document.exitFullscreen?.();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#08090d] text-white">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-white/20 border-t-iris-2" />
          <p className="text-sm font-medium text-white/60">Preparing live slideshow…</p>
        </div>
      </div>
    );
  }

  const baseUrl = process.env.REACT_APP_PUBLIC_URL || window.location.origin;
  const joinUrl = `${baseUrl.replace(/\/$/, '')}/attendjoin?code=${event?.event_code || ""}`;
  const currentPhoto = photos[currentIndex];

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-[#08090d] text-white select-none">
      {/* Top Floating Control Bar */}
      <div className="absolute top-4 left-4 right-4 z-40 flex items-center justify-between pointer-events-auto">
        <button
          onClick={() => navigate(`/events/${eventId}`)}
          className="flex items-center gap-2 rounded-xl border border-white/10 bg-black/40 backdrop-blur-md px-4 py-2 text-xs font-semibold text-white/80 hover:bg-black/60 transition-all"
        >
          <ArrowLeft className="h-4 w-4" /> Exit Slideshow
        </button>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying((prev) => !prev)}
            className="flex items-center gap-2 rounded-xl border border-white/10 bg-black/40 backdrop-blur-md px-4 py-2 text-xs font-semibold text-white/80 hover:bg-black/60 transition-all"
          >
            {isPlaying ? <Pause className="h-4 w-4 text-iris-2" /> : <Play className="h-4 w-4 text-match" />}
            {isPlaying ? "Pause" : "Play"}
          </button>
          <button
            onClick={toggleFullscreen}
            className="flex items-center gap-2 rounded-xl border border-white/10 bg-black/40 backdrop-blur-md px-4 py-2 text-xs font-semibold text-white/80 hover:bg-black/60 transition-all"
          >
            {isFullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
            {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          </button>
        </div>
      </div>

      {/* Main Photo Slideshow Stage */}
      {photos.length === 0 ? (
        <div className="flex h-full w-full flex-col items-center justify-center p-8 text-center">
          <QrCode className="mb-4 h-16 w-16 text-iris-2 animate-bounce" />
          <h2 className="text-3xl font-bold tracking-tight text-white mb-2">{event?.title}</h2>
          <p className="text-sm text-white/60 max-w-md">
            No photos uploaded yet. Scan the QR code in the corner to upload the first event photo!
          </p>
        </div>
      ) : (
        <AnimatePresence mode="wait">
          {currentPhoto && (
            <motion.div
              key={currentPhoto.id}
              initial={{ opacity: 0, scale: 1.05 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              className="absolute inset-0 flex items-center justify-center"
            >
              <img
                src={`${BACKEND_URL}/api/photos/${currentPhoto.id}`}
                alt="Event memory"
                className="h-full w-full object-contain p-4"
              />
            </motion.div>
          )}
        </AnimatePresence>
      )}

      {/* Bottom Corner Persistent QR Code Overlay for Venue Attendees */}
      <div className="absolute bottom-6 right-6 z-40 flex items-center gap-4 rounded-2xl border border-white/15 bg-black/75 backdrop-blur-xl p-4 shadow-2xl">
        <div className="p-2 bg-white rounded-xl">
          <QRCodeSVG value={joinUrl} size={90} level="H" />
        </div>
        <div>
          <p className="text-[0.65rem] font-mono uppercase tracking-[0.16em] text-iris-2 font-bold">
            Scan to Share & Find Photos
          </p>
          <h4 className="text-sm font-bold text-white mt-0.5 line-clamp-1 max-w-[160px]">
            {event?.title}
          </h4>
          <p className="text-xs text-white/60 font-mono mt-1">
            Code: <span className="text-match font-bold tracking-wider">{event?.event_code}</span>
          </p>
        </div>
      </div>
    </div>
  );
}
