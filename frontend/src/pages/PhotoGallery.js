import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft, Download, Camera, Image, CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';

export default function PhotoGallery() {
  const navigate = useNavigate();
  const location = useLocation();

  // ✅ REAL DATA (from PhotoGallery)
  const results = location.state?.results || [];
  const event = location.state?.event;

  const [selectedPhotos, setSelectedPhotos] = useState(new Set());

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

  const HIGH_CONFIDENCE_THRESHOLD = 0.7;

  const getConfidence = (score) => {
    return score >= HIGH_CONFIDENCE_THRESHOLD
      ? "High Confidence"
      : "Possible Match";
  };

  const toggleSelect = (id) => {
    const newSet = new Set(selectedPhotos);
    newSet.has(id) ? newSet.delete(id) : newSet.add(id);
    setSelectedPhotos(newSet);
  };

  const download = (photo) => {
    const url = `${BACKEND_URL}${photo.photo_url}`;
    window.open(url, '_blank');
  };

  const bulkDownload = () => {
    if (selectedPhotos.size === 0) {
      return toast.error("Select photos first");
    }

    selectedPhotos.forEach(id => {
      const photo = results.find(r => r.photo_id === id);
      if (photo) window.open(`${BACKEND_URL}${photo.photo_url}`);
    });

    toast.success(`Downloading ${selectedPhotos.size} photos`);
  };

  return (
    <div className="min-h-screen bg-slate-950">

      {/* HEADER (UI from PhotoResults) */}
      <div className="border-b border-white/5 bg-slate-900/20 backdrop-blur sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 py-4 flex justify-between items-center">
          <button onClick={() => navigate('/attendjoin')}
            className="flex items-center gap-2 text-slate-400 hover:text-white text-sm">
            <ArrowLeft className="w-4 h-4" /> New Search
          </button>

          <div className="flex items-center gap-2">
            <Camera className="text-white" />
            <span className="text-white font-bold">FaceShot</span>
          </div>

          {results.length > 0 && (
            <button
              onClick={bulkDownload}
              className="text-xs px-3 py-2 rounded-xl bg-indigo-500/10 text-indigo-300"
            >
              Download ({selectedPhotos.size})
            </button>
          )}
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-12">

        {/* TOP SECTION */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center mb-10">
          <div className={`w-20 h-20 mx-auto rounded-3xl flex items-center justify-center mb-4 ${
            results.length > 0
              ? 'bg-gradient-to-br from-emerald-500 to-teal-600'
              : 'bg-slate-700'
          }`}>
            {results.length > 0
              ? <CheckCircle className="text-white w-10 h-10" />
              : <Image className="text-gray-400 w-10 h-10" />}
          </div>

          <h1 className="text-3xl font-bold text-white">
            {results.length > 0
              ? `${results.length} Photos Found`
              : "No Photos Found"}
          </h1>

          {event && (
            <p className="text-slate-400 mt-2">{event.title}</p>
          )}
        </motion.div>

        {/* EMPTY STATE */}
        {results.length === 0 && (
          <div className="text-center">
            <button
              onClick={() => navigate('/attend')}
              className="bg-indigo-600 px-6 py-3 rounded-xl text-white"
            >
              Try Again
            </button>
          </div>
        )}

        {/* GRID (UI from PhotoResults + Logic from PhotoGallery) */}
        {results.length > 0 && (
          <>
            <div className="columns-2 md:columns-3 lg:columns-4 gap-3 space-y-3">

              {results.map((photo, i) => {
                const isSelected = selectedPhotos.has(photo.photo_id);

                return (
                  <motion.div
                    key={photo.photo_id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.05 }}
                    className="relative group rounded-xl overflow-hidden cursor-pointer break-inside-avoid"
                    onClick={() => toggleSelect(photo.photo_id)}
                  >
                    <img
                      src={`${BACKEND_URL}${photo.thumbnail_url}`}
                      className="w-full object-cover group-hover:scale-105 transition"
                    />

                    {/* CONFIDENCE (REAL LOGIC) */}
                    <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
                      {getConfidence(photo.similarity_score)}
                    </div>

                    {/* SELECTED */}
                    {isSelected && (
                      <div className="absolute top-2 right-2 bg-indigo-600 w-6 h-6 flex items-center justify-center rounded-full">
                        ✓
                      </div>
                    )}

                    {/* HOVER */}
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition">
                      <div className="absolute bottom-2 left-2 right-2 flex justify-between items-center text-white text-xs">
                        <span>
                          {(photo.similarity_score * 100).toFixed(0)}%
                        </span>

                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            download(photo);
                          }}
                          className="bg-white/20 p-2 rounded"
                        >
                          <Download size={14} />
                        </button>
                      </div>
                    </div>

                  </motion.div>
                );
              })}
            </div>

            {/* BULK DOWNLOAD */}
            <div className="text-center mt-10">
              <button
                onClick={bulkDownload}
                className="bg-indigo-600 px-8 py-4 rounded-xl text-white"
              >
                Download Selected ({selectedPhotos.size})
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
