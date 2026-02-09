import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from '@/components/ui/sonner';
import Landing from './pages/Landing';
import Auth from './pages/Auth';
import OrganizerDashboard from './pages/OrganizerDashboard';
import EventDetails from './pages/EventDetails';
import AttendeeEntry from './pages/AttendeeEntry';
import SelfieCapture from './pages/SelfieCapture';
import PhotoGallery from './pages/PhotoGallery';
import './App.css';

const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  return token ? children : <Navigate to="/auth" />;
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/auth" element={<Auth />} />
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <OrganizerDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/events/:eventId"
            element={
              <ProtectedRoute>
                <EventDetails />
              </ProtectedRoute>
            }
          />
          <Route path="/attend" element={<AttendeeEntry />} />
          <Route path="/attend/:eventId/selfie" element={<SelfieCapture />} />
          <Route path="/attend/:eventId/gallery" element={<PhotoGallery />} />
        </Routes>
      </BrowserRouter>
      <Toaster richColors position="top-right" />
    </div>
  );
}

export default App;
