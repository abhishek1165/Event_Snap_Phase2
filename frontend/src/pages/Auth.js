import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, ArrowRight, ArrowLeft } from 'lucide-react';
import { toast } from 'sonner';
import api from '@/utils/api';
import { cn } from '@/lib/utils';

const Auth = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('login'); // 'login' | 'signup'
  const [loginData, setLoginData] = useState({ email: '', password: '' });
  const [signupData, setSignupData] = useState({
    email: '',
    password: '',
    name: '',
    role: 'organizer'
  });

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await api.post('/auth/login', loginData);
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
      toast.success('Welcome back.');
      
      if (response.data.user.role === 'organizer') {
        navigate('/dashboard');
      } else {
        navigate('/attendjoin');
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Login failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await api.post('/auth/register', signupData);
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
      toast.success('Account created.');
      
      if (response.data.user.role === 'organizer') {
        navigate('/dashboard');
      } else {
        navigate('/attendjoin');
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Signup failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-50 flex flex-col items-center justify-center p-6 selection:bg-green-500/30">
      
      {/* Top Left Navigation */}
      <button 
        onClick={() => navigate('/')}
        className="absolute top-8 left-8 text-slate-400 hover:text-white transition-colors flex items-center gap-2 group text-sm font-medium tracking-wide"
      >
        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
        BACK
      </button>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-lg"
      >
        {/* Brand Header */}
        <div className="mb-16">
          <div className="inline-flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-green-500 rounded-lg flex items-center justify-center shadow-[0_0_20px_rgba(34,197,94,0.3)]">
              <Camera className="w-6 h-6 text-[#020617]" strokeWidth={2.5} />
            </div>
            <span className="font-bold text-2xl tracking-tight" style={{ fontFamily: 'Outfit, sans-serif' }}>
              FaceShot
            </span>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold tracking-tighter mb-4 leading-none" style={{ fontFamily: 'Outfit, sans-serif' }}>
            {activeTab === 'login' ? 'Sign in.' : 'Create.'}
          </h1>
          <p className="text-slate-400 text-lg md:text-xl font-light tracking-wide">
            {activeTab === 'login' ? 'Enter your details to proceed.' : 'Start managing your events today.'}
          </p>
        </div>

        {/* Custom Minimal Tabs */}
        <div className="flex items-center gap-8 mb-10 border-b border-slate-800 pb-2">
          <button
            onClick={() => setActiveTab('login')}
            className={cn(
              "text-lg font-medium tracking-wide transition-colors relative pb-2",
              activeTab === 'login' ? "text-white" : "text-slate-600 hover:text-slate-400"
            )}
          >
            Log In
            {activeTab === 'login' && (
              <motion.div layoutId="tab-indicator" className="absolute bottom-[-9px] left-0 right-0 h-[2px] bg-green-500" />
            )}
          </button>
          <button
            onClick={() => setActiveTab('signup')}
            className={cn(
              "text-lg font-medium tracking-wide transition-colors relative pb-2",
              activeTab === 'signup' ? "text-white" : "text-slate-600 hover:text-slate-400"
            )}
          >
            Sign Up
            {activeTab === 'signup' && (
              <motion.div layoutId="tab-indicator" className="absolute bottom-[-9px] left-0 right-0 h-[2px] bg-green-500" />
            )}
          </button>
        </div>

        {/* Forms Container */}
        <div className="relative">
          <AnimatePresence mode="wait">
            {activeTab === 'login' ? (
              <motion.form
                key="login-form"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.3 }}
                onSubmit={handleLogin}
                className="space-y-6"
              >
                <div className="space-y-2">
                  <label className="text-xs font-semibold tracking-widest text-slate-400 uppercase">Email Address</label>
                  <input
                    type="email"
                    value={loginData.email}
                    onChange={(e) => setLoginData({ ...loginData, email: e.target.value })}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-4 py-4 text-lg text-white placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500 transition-all"
                    placeholder="you@example.com"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-semibold tracking-widest text-slate-400 uppercase">Password</label>
                  <input
                    type="password"
                    value={loginData.password}
                    onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-4 py-4 text-lg text-white placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500 transition-all"
                    placeholder="••••••••"
                    required
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-white text-[#020617] font-bold text-lg rounded-lg px-4 py-4 mt-4 hover:bg-slate-200 transition-colors flex items-center justify-center gap-2 group disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {loading ? 'Processing...' : 'Continue'}
                  {!loading && <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />}
                </button>
              </motion.form>
            ) : (
              <motion.form
                key="signup-form"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.3 }}
                onSubmit={handleSignup}
                className="space-y-6"
              >
                <div className="space-y-2">
                  <label className="text-xs font-semibold tracking-widest text-slate-400 uppercase">Full Name</label>
                  <input
                    type="text"
                    value={signupData.name}
                    onChange={(e) => setSignupData({ ...signupData, name: e.target.value })}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-4 py-4 text-lg text-white placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500 transition-all"
                    placeholder="John Doe"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-semibold tracking-widest text-slate-400 uppercase">Email Address</label>
                  <input
                    type="email"
                    value={signupData.email}
                    onChange={(e) => setSignupData({ ...signupData, email: e.target.value })}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-4 py-4 text-lg text-white placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500 transition-all"
                    placeholder="you@example.com"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-semibold tracking-widest text-slate-400 uppercase">Password</label>
                  <input
                    type="password"
                    value={signupData.password}
                    onChange={(e) => setSignupData({ ...signupData, password: e.target.value })}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-4 py-4 text-lg text-white placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500 transition-all"
                    placeholder="••••••••"
                    required
                  />
                </div>
                
                {/* Segmented Control for Role */}
                <div className="space-y-3 pt-2">
                  <label className="text-xs font-semibold tracking-widest text-slate-400 uppercase">I am joining as a</label>
                  <div className="flex p-1 bg-slate-900 border border-slate-800 rounded-xl">
                    <button
                      type="button"
                      onClick={() => setSignupData({ ...signupData, role: 'organizer' })}
                      className={cn(
                        "flex-1 py-3 rounded-lg text-sm font-semibold transition-all",
                        signupData.role === 'organizer' 
                          ? "bg-slate-800 text-white shadow-sm" 
                          : "text-slate-500 hover:text-slate-300"
                      )}
                    >
                      Event Organizer
                    </button>
                    <button
                      type="button"
                      onClick={() => setSignupData({ ...signupData, role: 'attendee' })}
                      className={cn(
                        "flex-1 py-3 rounded-lg text-sm font-semibold transition-all",
                        signupData.role === 'attendee' 
                          ? "bg-slate-800 text-white shadow-sm" 
                          : "text-slate-500 hover:text-slate-300"
                      )}
                    >
                      Attendee
                    </button>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-green-500 text-[#020617] font-bold text-lg rounded-lg px-4 py-4 mt-6 hover:bg-green-400 transition-colors flex items-center justify-center gap-2 group disabled:opacity-70 disabled:cursor-not-allowed shadow-[0_0_30px_rgba(34,197,94,0.2)] hover:shadow-[0_0_40px_rgba(34,197,94,0.4)]"
                >
                  {loading ? 'Creating...' : 'Create Account'}
                  {!loading && <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />}
                </button>
              </motion.form>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
};

export default Auth;
