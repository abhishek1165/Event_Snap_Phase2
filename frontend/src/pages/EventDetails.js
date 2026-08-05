import React, { useState, useEffect, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Upload, CheckCircle, Loader2, Image as ImageIcon, Trash2, AlertTriangle, RefreshCw } from "lucide-react";
import { Button as ShadButton } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { toast } from "sonner";
import api from "@/utils/api";
import { AppShell, AppHeader } from "@/components/AppShell";
import { Button, Card } from "@/components/brand/atoms";
import { cn } from "@/lib/utils";

const EventDetails = () => {
  const { eventId } = useParams();
  const navigate = useNavigate();
  const [event, setEvent] = useState(null);
  const [status, setStatus] = useState(null);
  const [photos, setPhotos] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [loading, setLoading] = useState(true);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [photoToDelete, setPhotoToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);

  // preserved exactly — used to build thumbnail/photo URLs
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

  // ── API logic preserved verbatim ──
  const loadStatus = useCallback(async () => {
    try {
      const response = await api.get(`/events/${eventId}/status`);
      setStatus(response.data);
    } catch (error) {
      console.error("Failed to load status");
    }
  }, [eventId]);

  const loadEventDetails = useCallback(async () => {
    try {
      const response = await api.get(`/events/${eventId}`);
      setEvent(response.data);
      await loadStatus();
    } catch (error) {
      toast.error("Failed to load event");
    } finally {
      setLoading(false);
    }
  }, [eventId, loadStatus]);

  const loadPhotos = useCallback(async () => {
    try {
      const response = await api.get(`/events/${eventId}/photos`);
      setPhotos(response.data);
    } catch (error) {
      console.error("Failed to load photos");
    }
  }, [eventId]);

  // 3-second status poll — preserved
  useEffect(() => {
    loadEventDetails();
    loadPhotos();
    const interval = setInterval(loadStatus, 3000);
    return () => clearInterval(interval);
  }, [eventId, loadEventDetails, loadPhotos, loadStatus]);

  const handleDeletePhoto = async () => {
    if (!photoToDelete) return;
    setDeleting(true);
    try {
      await api.delete(`/events/${eventId}/photos/${photoToDelete.id}`);
      toast.success("Photo deleted successfully");
      setDeleteDialogOpen(false);
      setPhotoToDelete(null);
      await loadPhotos();
      await loadEventDetails();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to delete photo");
    } finally {
      setDeleting(false);
    }
  };

  // upload — FormData with repeated 'files' field, preserved
  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    setUploading(true);
    setUploadProgress(0);
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));
      await api.post(`/events/${eventId}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        },
      });
      toast.success(`${files.length} photos uploaded successfully!`);
      await loadPhotos();
      setTimeout(loadEventDetails, 1000);
    } catch (error) {
      console.error("Upload error:", error);
      toast.error(error.response?.data?.detail || error.message || "Upload failed");
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleReprocess = async () => {
    setReprocessing(true);
    try {
      await api.post(`/events/${eventId}/reprocess`);
      toast.success("Rebuilding face index. Try selfie search again when processing finishes.");
      await loadStatus();
      await loadEventDetails();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Reprocess failed");
    } finally {
      setReprocessing(false);
    }
  };

  if (loading) {
    return (
      <AppShell>
        <div className="flex min-h-[60dvh] items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <div className="h-10 w-10 rounded-full border-2 border-frame border-t-iris-2 animate-spin" />
            <p className="text-sm font-medium tracking-wide text-text-faint">Loading event…</p>
          </div>
        </div>
      </AppShell>
    );
  }

  const processingProgress = status ? (status.total_photos > 0 ? (status.processed_photos / status.total_photos) * 100 : 0) : 0;

  return (
    <AppShell
      header={
        <AppHeader
          onLogoClick={() => navigate("/dashboard")}
          right={
            <ShadButton
              data-testid="back-to-dashboard-button"
              variant="ghost"
              onClick={() => navigate("/dashboard")}
              className="gap-2 text-text-dim hover:text-text"
            >
              <ArrowLeft className="h-4 w-4" /> Dashboard
            </ShadButton>
          }
        />
      }
    >
      <main className="mx-auto max-w-7xl px-5 py-10 sm:px-6 lg:px-8">
        {/* event header */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="mb-8 p-6 sm:p-8">
            <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
              <div>
                <h1 className="mb-2 text-3xl font-semibold tracking-tight text-text">{event?.title}</h1>
                {event?.description && <p className="text-text-dim">{event.description}</p>}
              </div>
              <div className="text-right">
                <div className="mb-1 text-sm text-text-faint">Access code</div>
                <div className="font-mono text-2xl font-bold tracking-wider text-match">{event?.event_code}</div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 sm:gap-4">
              {[
                ["Total photos", status?.total_photos || 0],
                ["Processed", status?.processed_photos || 0],
                ["Faces detected", status?.faces_detected || 0],
              ].map(([label, val]) => (
                <div key={label} className="rounded-xl border border-frame-soft bg-surface-2 p-4">
                  <div className="mb-1 text-2xl font-bold tnum text-text">{val}</div>
                  <div className="text-xs text-text-faint sm:text-sm">{label}</div>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>

        {/* processing status — aperture "developing" viz */}
        {status && status.status === "processing" && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 rounded-xl border border-memory/30 bg-memory/[0.06] p-6"
          >
            <div className="mb-4 flex items-center gap-3">
              <span className="relative flex h-5 w-5 items-center justify-center">
                <span className="absolute inline-flex h-full w-full rounded-full border-2 border-memory/40 border-t-memory animate-spin" />
              </span>
              <span className="font-semibold text-memory">Processing photos…</span>
            </div>
            <Progress value={processingProgress} className="h-2" />
            <div className="mt-2 text-sm text-memory/80 tnum">
              {status.processed_photos} of {status.total_photos} photos processed
            </div>
          </motion.div>
        )}

        {status && status.status === "completed" && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 rounded-xl border border-match/30 bg-match/[0.06] p-6"
          >
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <CheckCircle className="h-5 w-5 text-match" />
                <span className="font-semibold text-match">All photos processed. Event is ready for attendees.</span>
              </div>
              <Button
                variant="subtle"
                size="sm"
                onClick={handleReprocess}
                disabled={reprocessing || !photos.length}
                className="gap-2"
              >
                {reprocessing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                Rebuild face index
              </Button>
            </div>
            <p className="mt-2 text-sm text-match/70">
              If selfie search does not find you, rebuild the face index to re-detect faces. No need to re-upload.
            </p>
          </motion.div>
        )}

        {/* upload */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <Card className="p-6 sm:p-8">
            <h2 className="mb-4 text-2xl font-semibold tracking-tight text-text">Upload photos</h2>
            <p className="mb-6 text-text-dim">Select multiple photos. Face detection and indexing start automatically.</p>

            <div className="rounded-xl border-2 border-dashed border-frame p-12 text-center transition-colors hover:border-iris-2">
              <input
                data-testid="photo-upload-input"
                type="file"
                multiple
                accept="image/*"
                onChange={handleFileUpload}
                disabled={uploading}
                className="hidden"
                id="photo-upload"
              />
              <label htmlFor="photo-upload" className="cursor-pointer">
                {uploading ? (
                  <div>
                    <Loader2 className="mx-auto mb-4 h-12 w-12 animate-spin text-iris-2" />
                    <p className="mb-2 font-semibold text-text">Uploading…</p>
                    <Progress value={uploadProgress} className="mx-auto h-2 w-64" />
                    <p className="mt-2 text-sm text-text-faint tnum">{uploadProgress}%</p>
                  </div>
                ) : (
                  <div>
                    <Upload className="mx-auto mb-4 h-12 w-12 text-text-faint" />
                    <p className="mb-2 font-semibold text-text">Click to upload photos</p>
                    <p className="text-sm text-text-faint">or drag and drop multiple images</p>
                  </div>
                )}
              </label>
            </div>
          </Card>
        </motion.div>

        {/* photos grid */}
        {photos.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mt-8"
          >
            <Card className="p-6 sm:p-8">
              <h2 className="mb-6 text-2xl font-semibold tracking-tight text-text">
                Uploaded photos <span className="text-text-faint tnum">({photos.length})</span>
              </h2>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
                {photos.map((photo, index) => (
                  <motion.div
                    key={photo.id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="group relative aspect-square overflow-hidden rounded-lg border border-frame-soft transition-colors hover:border-iris-2"
                  >
                    <img
                      src={`${BACKEND_URL}/api/photos/${photo.id}/thumbnail`}
                      alt={`Photo ${index + 1}`}
                      loading="lazy"
                      className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                    />
                    <div className="absolute inset-0 flex items-center justify-center bg-ink/50 opacity-0 transition-opacity group-hover:opacity-100">
                      <span className="text-xs font-medium text-text">
                        {photo.faces_detected} face{photo.faces_detected !== 1 ? "s" : ""}
                      </span>
                    </div>
                    <div className="absolute right-2 top-2 opacity-0 transition-opacity group-hover:opacity-100">
                      <button
                        aria-label="Delete photo"
                        onClick={(e) => {
                          e.stopPropagation();
                          setPhotoToDelete(photo);
                          setDeleteDialogOpen(true);
                        }}
                        className="flex h-8 w-8 items-center justify-center rounded-full bg-red-500/90 text-white shadow-lg transition-transform hover:scale-105 active:scale-95"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>
        )}

        {/* delete dialog */}
        <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
          <DialogContent className="border-frame bg-surface text-text">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-red-400">
                <AlertTriangle className="h-5 w-5" /> Delete photo
              </DialogTitle>
              <DialogDescription className="text-text-dim">
                Delete this photo permanently? This removes it from storage, the database, the face recognition index, and all search results. This cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <div className="mt-4 flex gap-3">
              <Button
                variant="ghost"
                onClick={() => {
                  setDeleteDialogOpen(false);
                  setPhotoToDelete(null);
                }}
                className="flex-1"
                disabled={deleting}
              >
                Cancel
              </Button>
              <button
                onClick={handleDeletePhoto}
                disabled={deleting}
                className={cn(
                  "flex flex-1 items-center justify-center gap-2 rounded-xl bg-red-500 px-4 py-3 text-sm font-semibold text-white transition-transform hover:bg-red-400 active:scale-[.98] disabled:opacity-50"
                )}
              >
                {deleting ? (<><Loader2 className="h-4 w-4 animate-spin" /> Deleting…</>) : (<><Trash2 className="h-4 w-4" /> Delete photo</>)}
              </button>
            </div>
          </DialogContent>
        </Dialog>
      </main>
    </AppShell>
  );
};

export default EventDetails;
