import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Webcam from 'react-webcam';
import { Check, AlertCircle, Upload, Lightbulb, Monitor, Smartphone } from 'lucide-react';
import { AppShell, AppHeader } from '@/components/AppShell';
import { Button, Eyebrow, Card, Badge } from '@/components/brand/atoms';
import { T, MONO, DISPLAY, SHADOW, DUR, EASE } from '@/design/tokens';

import { toast } from 'sonner';
import api from '@/utils/api';

const SelfieCapture = () => {
  const { eventId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const event = location.state?.event;
  const webcamRef = useRef(null);
  const [imgSrc, setImgSrc] = useState(null);
  const [searching, setSearching] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [uploadMode, setUploadMode] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const fileInputRef = useRef(null);
  const DeviceIcon = isMobile ? Smartphone : Monitor;

  useEffect(() => {
    const mobileCheck = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    setIsMobile(mobileCheck);
  }, []);

  const videoConstraints = {
    facingMode: { exact: 'user' },
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 },
  };

  const handleCameraError = (error) => {
    console.error('Camera error:', error);
    setCameraError(true);
    toast.error('Failed to access camera. Please ensure camera permissions are granted.');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.match('image.*')) {
      toast.error('Please select an image file');
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      setImgSrc(ev.target.result);
      setUploadedFile(file);
      setUploadMode(true);
    };
    reader.readAsDataURL(file);
  };

  const handleRetake = () => {
    setImgSrc(null);
    setUploadedFile(null);
    setUploadMode(false);
  };

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) setImgSrc(imageSrc);
  }, [webcamRef]);

  const handleSearch = async () => {
    if (!imgSrc) return;
    setSearching(true);
    try {
      let blob;
      if (uploadedFile) {
        blob = uploadedFile;
      } else {
        const response = await fetch(imgSrc);
        blob = await response.blob();
      }

      const formData = new FormData();
      formData.append('file', blob, 'selfie.jpg');
      formData.append('event_id', eventId);

      const searchResponse = await api.post('/search/selfie', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const results = searchResponse.data;

      if (results.length === 0) {
        toast.error('No photos found. Try a different selfie.');
        setImgSrc(null);
      } else {
        toast.success(`Found ${results.length} photos!`);
        navigate(`/attend/${eventId}/gallery`, { state: { results, event } });
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Search failed. Please try again.';
      if (error.response?.status === 404 && errorMessage.includes('not present')) {
        toast.error('No matching photos found. You are not present in this event.');
      } else {
        toast.error(errorMessage);
      }
      setImgSrc(null);
    } finally {
      setSearching(false);
    }
  };

  /* ── Framer Motion page entrance ── */
  const pageVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: DUR.base, ease: EASE.out } },
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 16 },
    visible: { opacity: 1, y: 0, transition: { duration: DUR.fast, ease: EASE.out } },
  };

  return (
    <AppShell
      header={
        <AppHeader
          showLogo={false}
          right={
            <Badge tone="iris">{isMobile ? 'Mobile' : 'Desktop'} camera</Badge>
          }
        >
          <Button
            data-testid="back-button"
            variant="ghost"
            size="sm"
            onClick={() => navigate('/attendjoin')}
            className="gap-2"
          >
            ← Back
          </Button>
        </AppHeader>
      }
    >
      <motion.div
        variants={pageVariants}
        initial="hidden"
        animate="visible"
        className="mx-auto w-full max-w-2xl px-5 py-8 sm:py-12"
      >
        {/* ── Page header ── */}
        <div className="mb-8 text-center">
          <Eyebrow className="mb-3">Selfie capture</Eyebrow>
          <h1
            className="text-2xl font-bold tracking-tight sm:text-3xl"
            style={{ fontFamily: DISPLAY }}
          >
            {event?.title || 'Find your photos'}
          </h1>
          <p className="mt-2 text-sm text-text-dim">
            One clear face photo is all we need. We turn it into a match key, never a photo we keep.
          </p>
        </div>

        {/* ── Capture card ── */}
        <motion.div variants={cardVariants}>
          <Card className="overflow-hidden">
            {/* ── Mode tabs ── */}
            <div className="flex border-b border-frame-soft px-4 pt-3">
              {['selfie', 'upload'].map((mode) => (
                <button
                  key={mode}
                  onClick={() => {
                    setUploadMode(mode === 'upload');
                    if (mode === 'upload' && !imgSrc) {
                      setImgSrc(null);
                      setUploadedFile(null);
                    }
                  }}
                  className={`relative flex-1 rounded-t-lg px-4 pb-3 text-center text-sm font-medium transition-colors ${
                    uploadMode === (mode === 'upload')
                      ? 'text-text'
                      : 'text-text-dim hover:text-text'
                  }`}
                >
                  {mode === 'selfie' ? 'Take Selfie' : 'Upload Photo'}
                  {uploadMode === (mode === 'upload') && (
                    <motion.div
                      layoutId="capture-tab-pill"
                      className="absolute inset-x-4 bottom-0 h-0.5 bg-iris-2 rounded-full"
                      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                    />
                  )}
                </button>
              ))}
            </div>

            {/* ── Viewfinder ── */}
            <div className="relative aspect-[4/3] bg-ink overflow-hidden">
              <AnimatePresence mode="wait">
                {!imgSrc ? (
                  <motion.div
                    key={uploadMode ? 'upload' : 'camera'}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0"
                  >
                    {uploadMode ? (
                      /* ── Upload zone ── */
                      <div className="flex h-full items-center justify-center p-6">
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="image/*"
                          onChange={handleFileUpload}
                          className="hidden"
                          id="selfie-file-upload"
                        />
                        <label
                          htmlFor="selfie-file-upload"
                          className="flex h-full w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-frame hover:border-iris-2 transition-colors p-8 text-center"
                        >
                          <Upload className="mx-auto mb-3 h-10 w-10 text-text-faint" />
                          <p className="mb-1 text-sm font-semibold text-text">
                            Tap to upload a photo
                          </p>
                          <p className="text-xs text-text-dim">
                            Supports JPG, PNG, WEBP
                          </p>
                        </label>
                      </div>
                    ) : cameraError ? (
                      /* ── Camera error ── */
                      <div className="flex h-full flex-col items-center justify-center gap-3 bg-surface-2 p-6 text-center">
                        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-destructive/10">
                          <AlertCircle className="h-7 w-7 text-destructive" />
                        </div>
                        <h3 className="text-sm font-semibold text-text">
                          Camera Access Denied
                        </h3>
                        <p className="text-xs text-text-dim max-w-[260px]">
                          Please allow camera access in your device settings to capture your selfie.
                        </p>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            setCameraError(false);
                            if (webcamRef.current) webcamRef.current.video = null;
                          }}
                        >
                          Try Again
                        </Button>
                      </div>
                    ) : (
                      /* ── Live webcam ── */
                      <>
                        <Webcam
                          audio={false}
                          ref={webcamRef}
                          screenshotFormat="image/jpeg"
                          className="h-full w-full object-cover"
                          videoConstraints={videoConstraints}
                          onUserMediaError={handleCameraError}
                        />
                        {/* Aperture brackets overlay */}
                        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                          <div className="relative" style={{ width: '38%', paddingBottom: '48%' }}>
                            {[
                              { top: 0, left: 0, borderRight: 'none', borderBottom: 'none' },
                              { top: 0, right: 0, borderLeft: 'none', borderBottom: 'none' },
                              { bottom: 0, left: 0, borderRight: 'none', borderTop: 'none' },
                              { bottom: 0, right: 0, borderLeft: 'none', borderTop: 'none' },
                            ].map((s, i) => (
                              <span
                                key={i}
                                className="absolute transition-colors duration-500"
                                style={{
                                  width: 22,
                                  height: 22,
                                  border: '2.5px solid rgba(139,123,255,.6)',
                                  ...s,
                                }}
                              />
                            ))}
                          </div>
                        </div>
                        {/* Scan line */}
                        <motion.div
                          animate={{ top: ['35%', '65%', '35%'] }}
                          transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
                          className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-iris-2 to-transparent opacity-70"
                          style={{ boxShadow: '0 0 12px 2px rgba(109,94,245,.35)' }}
                        />
                      </>
                    )}
                  </motion.div>
                ) : (
                  /* ── Preview ── */
                  <motion.img
                    key="preview"
                    initial={{ opacity: 0, scale: 1.02 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    src={imgSrc}
                    alt="Selfie preview"
                    className="h-full w-full object-cover"
                  />
                )}
              </AnimatePresence>
            </div>

            {/* ── Controls ── */}
            <div className="flex items-center justify-center gap-4 border-t border-frame-soft bg-surface p-5">
              {!imgSrc ? (
                <>
                  {!uploadMode && (
                    <button
                      data-testid="capture-button"
                      onClick={capture}
                      className="group relative flex h-16 w-16 items-center justify-center rounded-full border-[3px] border-iris-2 bg-transparent transition-all hover:border-iris"
                      aria-label="Capture selfie"
                    >
                      <span className="block h-11 w-11 rounded-full bg-iris-2 transition-all group-hover:bg-iris group-active:scale-95" />
                    </button>
                  )}
                  <p className="text-xs text-text-dim">
                    {!uploadMode ? 'Tap the shutter to capture' : 'Select a photo to continue'}
                  </p>
                </>
              ) : (
                <div className="flex w-full gap-3">
                  <Button
                    data-testid="retake-button"
                    variant="ghost"
                    onClick={handleRetake}
                    disabled={searching}
                    className="flex-1"
                  >
                    Retake
                  </Button>
                  <Button
                    data-testid="search-button"
                    variant="primary"
                    onClick={handleSearch}
                    disabled={searching}
                    className="flex-1 gap-2"
                  >
                    {searching ? (
                      <>
                        <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                        Searching…
                      </>
                    ) : (
                      <>
                        <Check className="h-4 w-4" />
                        Search Photos
                      </>
                    )}
                  </Button>
                </div>
              )}
            </div>
          </Card>
        </motion.div>

        {/* ── Tips card ── */}
        <motion.div variants={cardVariants} className="mt-6">
          <Card className="p-5">
            <div className="mb-3 flex items-center gap-2">
              <Lightbulb className="h-4 w-4 text-memory" />
              <span className="text-sm font-semibold text-text">Tips for best results</span>
            </div>
            <ul className="space-y-2">
              {[
                'Face the camera directly',
                'Make sure your face is well-lit',
                'Remove sunglasses or masks if possible',
                'Keep a neutral expression similar to event photos',
              ].map((tip) => (
                <li key={tip} className="flex items-start gap-2 text-xs text-text-dim">
                  <span className="mt-1 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-iris-2" />
                  {tip}
                </li>
              ))}
            </ul>
          </Card>
        </motion.div>
      </motion.div>
    </AppShell>
  );
};

export default SelfieCapture;
