import React, { useMemo, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Download, Check, CheckCircle, Search, ImageIcon } from 'lucide-react';
import { AppShell, AppHeader } from '@/components/AppShell';
import { Button, Badge, Eyebrow } from '@/components/brand/atoms';
import { getBackendUrl } from '@/utils/api';
import { T, DISPLAY, SHADOW, DUR, EASE } from '@/design/tokens';
import { toast } from 'sonner';

const EMPTY_RESULTS = [];

export default function PhotoGallery() {
  const navigate = useNavigate();
  const location = useLocation();

  const results = location.state?.results ?? EMPTY_RESULTS;
  const event = location.state?.event;

  const [selectedPhotos, setSelectedPhotos] = useState(new Set());

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
  const HIGH_CONFIDENCE_THRESHOLD = 0.7;

  const resultsById = useMemo(
    () => new Map(results.map((photo) => [photo.photo_id, photo])),
    [results]
  );

  const getConfidence = (score) => {
    return score >= HIGH_CONFIDENCE_THRESHOLD ? 'High Confidence' : 'Possible Match';
  };

  const getConfidenceTone = (score) => {
    return score >= HIGH_CONFIDENCE_THRESHOLD ? 'match' : 'amber';
  };

  const guessFilenameFromUrl = (url, fallback) => {
    try {
      const last = new URL(url).pathname.split('/').filter(Boolean).pop();
      return last || fallback;
    } catch {
      const parts = String(url).split('?')[0].split('/').filter(Boolean);
      return parts[parts.length - 1] || fallback;
    }
  };

  const downloadUrlAsFile = async (url, filename) => {
    const res = await fetch(url, { credentials: 'include' });
    if (!res.ok) throw new Error(`Download failed (${res.status})`);
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    try {
      const a = document.createElement('a');
      a.href = objectUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  };

  const toggleSelect = (id) => {
    setSelectedPhotos((current) => {
      const next = new Set(current);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const download = async (photo) => {
    try {
      const url = `${BACKEND_URL}${photo.photo_url}`;
      const filename = guessFilenameFromUrl(url, `photo-${photo.photo_id}.jpg`);
      await downloadUrlAsFile(url, filename);
    } catch (err) {
      toast.error(err?.message || 'Download failed');
    }
  };

  const bulkDownload = async () => {
    if (selectedPhotos.size === 0) return toast.error('Select photos first');

    const photos = Array.from(selectedPhotos)
      .map((id) => resultsById.get(id))
      .filter(Boolean);

    let ok = 0;
    for (const photo of photos) {
      try {
        const url = `${BACKEND_URL}${photo.photo_url}`;
        const filename = guessFilenameFromUrl(url, `photo-${photo.photo_id}.jpg`);
        await downloadUrlAsFile(url, filename);
        ok += 1;
      } catch {
        // per-photo failure tracked below
      }
    }

    if (ok === photos.length) toast.success(`Downloading ${ok} photos`);
    else if (ok > 0) toast.success(`Downloaded ${ok}/${photos.length} photos`);
    else toast.error('Could not download selected photos');
  };

  /* ── Animation variants (Framer Motion for app UI) ── */
  const pageVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: DUR.base, ease: EASE.out } },
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.04, delayChildren: 0.1 },
    },
  };

  const cardVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { opacity: 1, scale: 1, transition: { type: 'spring', stiffness: 180, damping: 22 } },
  };

  return (
    <AppShell
      header={
        <AppHeader
          showLogo={false}
          right={
            results.length > 0 && (
              <Button
                variant="subtle"
                size="sm"
                onClick={bulkDownload}
                disabled={selectedPhotos.size === 0}
                className="gap-2"
                aria-label={`Download ${selectedPhotos.size} selected photos`}
              >
                <Download className="h-3.5 w-3.5" />
                {selectedPhotos.size > 0 && (
                  <span className="font-mono text-xs tabular-nums">{selectedPhotos.size}</span>
                )}
              </Button>
            )
          }
        >
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/attendjoin')}
            className="gap-2"
            aria-label="Start a new photo search"
          >
            <ArrowLeft className="h-4 w-4" />
            <span className="hidden sm:inline">New Search</span>
          </Button>
        </AppHeader>
      }
    >
      <motion.div
        variants={pageVariants}
        initial="hidden"
        animate="visible"
        className="mx-auto w-full max-w-5xl px-5 py-8 sm:py-12"
      >
        {/* ── Hero result summary ── */}
        <div className="mb-10 text-center">
          <div
            className={`mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl ${
              results.length > 0
                ? 'bg-match/15'
                : 'bg-surface-3'
            }`}
          >
            {results.length > 0 ? (
              <CheckCircle className="h-8 w-8 text-match" />
            ) : (
              <ImageIcon className="h-8 w-8 text-text-faint" />
            )}
          </div>

          <h1
            className="text-2xl font-bold tracking-tight sm:text-3xl"
            style={{ fontFamily: DISPLAY }}
          >
            {results.length > 0
              ? `${results.length} Photo${results.length !== 1 ? 's' : ''} Found`
              : 'No Photos Found'}
          </h1>

          {event && (
            <p className="mt-1.5 text-sm text-text-dim">{event.title}</p>
          )}

          {results.length > 0 && (
            <p className="mx-auto mt-3 max-w-md text-sm text-text-dim">
              Select the photos you want to keep, then download them together or one at a time.
            </p>
          )}
        </div>

        {/* ── Empty state ── */}
        {results.length === 0 && (
          <div className="mx-auto max-w-md rounded-2xl border border-frame-soft bg-surface p-6 text-center">
            <Search className="mx-auto mb-3 h-7 w-7 text-iris-2" />
            <p className="mb-5 text-sm text-text-dim">
              Try another selfie with your face centered and well lit, or ask the organizer to rebuild the face index.
            </p>
            <Button
              variant="primary"
              onClick={() => navigate('/attendjoin')}
            >
              Try Again
            </Button>
          </div>
        )}

        {/* ── Photo grid ── */}
        {results.length > 0 && (
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4"
          >
            {results.map((photo) => {
              const isSelected = selectedPhotos.has(photo.photo_id);
              const confidenceTone = getConfidenceTone(photo.similarity_score);

              return (
                <motion.div
                  key={photo.photo_id}
                  variants={cardVariants}
                  className={`group relative cursor-pointer overflow-hidden rounded-xl border bg-ink aspect-[4/5] transition-colors focus-within:ring-2 focus-within:ring-iris-2 ${
                    isSelected ? 'border-iris-2' : 'border-frame-soft'
                  }`}
                >
                  <img
                    src={`${BACKEND_URL}${photo.thumbnail_url}`}
                    alt={`${getConfidence(photo.similarity_score)} event photo match`}
                    loading="lazy"
                    decoding="async"
                    className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />

                  {/* Selection overlay */}
                  <button
                    type="button"
                    className="absolute inset-0 z-10 cursor-pointer"
                    onClick={() => toggleSelect(photo.photo_id)}
                    aria-pressed={isSelected}
                    aria-label={`${isSelected ? 'Deselect' : 'Select'} photo with ${getConfidence(photo.similarity_score).toLowerCase()}`}
                  />

                  {/* Confidence badge */}
                  <div className="pointer-events-none absolute top-2 left-2 z-20">
                    <Badge tone={confidenceTone}>
                      {getConfidence(photo.similarity_score)}
                    </Badge>
                  </div>

                  {/* Selected check */}
                  {isSelected && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="pointer-events-none absolute top-2 right-2 z-20 flex h-6 w-6 items-center justify-center rounded-full bg-iris"
                    >
                      <Check className="h-3.5 w-3.5 text-white" />
                    </motion.div>
                  )}

                  {/* Hover overlay */}
                  <div className="pointer-events-none absolute inset-0 z-20 bg-black/60 opacity-0 transition-opacity group-hover:opacity-100">
                    <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between text-xs text-white">
                      <span className="font-mono tabular-nums">
                        {(photo.similarity_score * 100).toFixed(0)}%
                      </span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          download(photo);
                        }}
                        className="pointer-events-auto flex h-8 w-8 items-center justify-center rounded-lg bg-white/20 transition-colors hover:bg-white/30"
                        aria-label="Download this photo"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        )}

        {/* ── Bulk download footer ── */}
        {results.length > 0 && (
          <div className="mt-10 text-center">
            <Button
              variant="primary"
              size="lg"
              onClick={bulkDownload}
              disabled={selectedPhotos.size === 0}
              className="gap-2"
            >
              <Download className="h-4 w-4" />
              Download Selected
              {selectedPhotos.size > 0 && (
                <span className="font-mono tabular-nums">({selectedPhotos.size})</span>
              )}
            </Button>
          </div>
        )}
      </motion.div>
    </AppShell>
  );
}
