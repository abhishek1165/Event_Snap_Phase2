import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from '@/components/ui/sonner';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import OrganizerDashboard from './pages/OrganizerDashboard';
import EventDetails from './pages/EventDetails';
import AttendeeEntry from './pages/AttendeeEntry';
import SelfieCapture from './pages/SelfieCapture';
import PhotoGallery from './pages/PhotoGallery';
import LiveSlideshow from './pages/LiveSlideshow';
import './App.css';

const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  return token ? children : <Navigate to="/login" />;
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/auth" element={<Navigate to="/login" replace />} />
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
          <Route
            path="/events/:eventId/slideshow"
            element={
              <ProtectedRoute>
                <LiveSlideshow />
              </ProtectedRoute>
            }
          />
          <Route path="/attend" element={<Navigate to="/attendjoin" replace />} />
          <Route path="/attendjoin" element={<AttendeeEntry />} />
          <Route path="/attend/:eventId/selfie" element={<SelfieCapture />} />
          <Route path="/attend/:eventId/gallery" element={<PhotoGallery />} />
        </Routes>
      </BrowserRouter>
      <Toaster richColors position="top-right" />
    </div>
  );
}

export default App;
