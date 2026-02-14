import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import Webcam from 'react-webcam';
import { Camera, Check, Loader2, ArrowLeft, AlertCircle, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
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
  const [uploadMode, setUploadMode] = useState(false); // Toggle between camera and upload
  const [uploadedFile, setUploadedFile] = useState(null);

  // Detect if user is on mobile device
  useEffect(() => {
    const mobileCheck = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    setIsMobile(mobileCheck);
  }, []);

  // Camera constraints - force use of current device camera
  const videoConstraints = {
    facingMode: { exact: "user" }, // Always use front/user-facing camera regardless of device
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 }
  };

  const handleCameraError = (error) => {
    console.error('Camera error:', error);
    setCameraError(true);
    toast.error('Failed to access camera. Please ensure you\'re using this device\'s camera.');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.match('image.*')) {
      toast.error('Please select an image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      setImgSrc(event.target.result);
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
    const imageSrc = webcamRef.current.getScreenshot();
    setImgSrc(imageSrc);
  }, [webcamRef]);

  const retake = () => {
    setImgSrc(null);
  };

  const handleSearch = async () => {
    if (!imgSrc) return;
    
    setSearching(true);
    try {
      let blob;
      if (uploadedFile) {
        // Use the uploaded file directly
        blob = uploadedFile;
      } else {
        // Convert captured image to blob
        const response = await fetch(imgSrc);
        blob = await response.blob();
      }
      
      const formData = new FormData();
      formData.append('file', blob, 'selfie.jpg');
      formData.append('event_id', eventId);
      
      const searchResponse = await api.post('/search/selfie', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
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
      // Handle the specific "no matching photos" case
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

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <Button
            data-testid="back-button"
            variant="ghost"
            onClick={() => navigate('/attend')}
            className="gap-2 mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </Button>
          <h1 className="text-3xl font-bold mb-2" style={{ fontFamily: 'Outfit, sans-serif' }}>
            Take Your Selfie
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            {event?.title || 'Event'} • Look at the camera and smile!
          </p>
        </motion.div>

        {/* Camera/Preview */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-xl"
        >
          {/* Mode Toggle */}
          <div className="flex border-b border-slate-200 dark:border-slate-700">
            <button
              className={`flex-1 py-3 text-center font-medium ${
                !uploadMode 
                  ? 'text-indigo-600 border-b-2 border-indigo-600' 
                  : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
              onClick={() => setUploadMode(false)}
            >
              Take Selfie
            </button>
            <button
              className={`flex-1 py-3 text-center font-medium ${
                uploadMode 
                  ? 'text-indigo-600 border-b-2 border-indigo-600' 
                  : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
              onClick={() => setUploadMode(true)}
            >
              Upload Photo
            </button>
          </div>
          
          <div className="relative aspect-[4/3] bg-slate-900">
            {!imgSrc ? (
              <>
                {uploadMode ? (
                  // Upload mode UI
                  <div className="w-full h-full flex items-center justify-center p-4">
                    <div className="w-full h-full flex flex-col items-center justify-center">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileUpload}
                        className="hidden"
                        id="photo-upload"
                      />
                      <label 
                        htmlFor="photo-upload"
                        className="cursor-pointer flex flex-col items-center justify-center w-full h-full border-2 border-dashed border-slate-400 rounded-xl hover:border-indigo-500 transition-colors p-8 text-center"
                      >
                        <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                        <p className="font-semibold mb-2 text-slate-700 dark:text-slate-300">Click to upload a photo</p>
                        <p className="text-sm text-slate-500 dark:text-slate-400">Supports JPG, PNG, WEBP</p>
                      </label>
                    </div>
                  </div>
                ) : (
                  // Camera mode UI
                  <>
                    {cameraError ? (
                      <div className="w-full h-full flex items-center justify-center bg-slate-100 dark:bg-slate-800">
                        <div className="text-center p-6">
                          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                          <h3 className="text-lg font-semibold mb-2 text-slate-800 dark:text-slate-200">Camera Access Denied</h3>
                          <p className="text-slate-600 dark:text-slate-400 mb-4">
                            Please allow camera access for this device to capture your selfie.
                          </p>
                          <Button 
                            onClick={() => {
                              setCameraError(false);
                              // Try to re-initialize camera
                              if (webcamRef.current) {
                                webcamRef.current.video = null;
                              }
                            }}
                            className="bg-indigo-600 hover:bg-indigo-700 text-white"
                          >
                            Try Again
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        className="w-full h-full object-cover"
                        videoConstraints={videoConstraints}
                        onUserMediaError={handleCameraError}
                      />
                    )}
                    {/* Scanning overlay animation */}
                    <motion.div
                      animate={{
                        y: ['0%', '100%', '0%']
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: 'linear'
                      }}
                      className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-indigo-500 to-transparent"
                      style={{ top: 0 }}
                    />
                  </>
                )}
              </>
            ) : (
              <img src={imgSrc} alt="Selfie preview" className="w-full h-full object-cover" />
            )}
          </div>

          {/* Controls */}
          <div className="p-6 bg-white dark:bg-slate-800">
            {!imgSrc ? (
              <div className="text-center">
                {!uploadMode && (
                  <Button
                    data-testid="capture-button"
                    onClick={capture}
                    size="lg"
                    className="rounded-full w-16 h-16 bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg p-0"
                  >
                    <Camera className="w-8 h-8" />
                  </Button>
                )}
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-4">
                  {!uploadMode ? 'Click the button to capture' : 'Select a photo to upload'}
                </p>
              </div>
            ) : (
              <div className="flex gap-3">
                <Button
                  data-testid="retake-button"
                  onClick={handleRetake}
                  variant="outline"
                  className="flex-1 h-12 rounded-lg"
                  disabled={searching}
                >
                  Retake
                </Button>
                <Button
                  data-testid="search-button"
                  onClick={handleSearch}
                  className="flex-1 h-12 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white gap-2"
                  disabled={searching}
                >
                  {searching ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Searching...
                    </>
                  ) : (
                    <>
                      <Check className="w-4 h-4" />
                      Search Photos
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>
        </motion.div>

        {/* Device Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mt-8 p-6 bg-blue-50 dark:bg-blue-900/10 rounded-xl border border-blue-200 dark:border-blue-800"
        >
          <h3 className="font-semibold mb-3 text-blue-900 dark:text-blue-300">
            {isMobile ? '📱 Mobile Device Detected' : '💻 Desktop/Laptop Detected'}
          </h3>
          <p className="text-sm text-blue-800 dark:text-blue-300">
            Using {isMobile ? 'your mobile device\'s front camera' : 'this device\'s webcam'} for selfie capture.
          </p>
        </motion.div>

        {/* Tips */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-4 p-6 bg-indigo-50 dark:bg-indigo-900/10 rounded-xl border border-indigo-200 dark:border-indigo-800"
        >
          <h3 className="font-semibold mb-3 text-indigo-900 dark:text-indigo-300">Tips for best results:</h3>
          <ul className="space-y-2 text-sm text-indigo-800 dark:text-indigo-300">
            <li>• Face the camera directly</li>
            <li>• Make sure your face is well-lit</li>
            <li>• Remove sunglasses or masks if possible</li>
            <li>• Keep a neutral expression similar to event photos</li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
};

export default SelfieCapture;
