import React from 'react';
import { Camera } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function LandingFooter() {
  return (
    <footer className="bg-slate-950 border-t border-white/5 py-12">
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <Link to="/" className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <Camera className="w-4 h-4 text-white" />
            </div>
            <span className="text-white font-bold">FaceShot</span>
          </Link>
          <p className="text-slate-600 text-sm">Privacy-first face recognition for events</p>
          <p className="text-slate-700 text-sm">© 2026 FaceShot. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}
