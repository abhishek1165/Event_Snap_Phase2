import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Camera, Mail, Lock, User, UserCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';
import api from '@/utils/api';

const Auth = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
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
      toast.success('Welcome back!');
      
      if (response.data.user.role === 'organizer') {
        navigate('/dashboard');
      } else {
        navigate('/attend');
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Login failed');
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
      toast.success('Account created successfully!');
      
      if (response.data.user.role === 'organizer') {
        navigate('/dashboard');
      } else {
        navigate('/attend');
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Signup failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-slate-50 dark:bg-slate-900">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-4">
            <Camera className="w-8 h-8 text-indigo-600" />
            <span className="font-bold text-2xl" style={{ fontFamily: 'Outfit, sans-serif' }}>FaceShot</span>
          </div>
          <h1 className="text-3xl font-bold mb-2" style={{ fontFamily: 'Outfit, sans-serif' }}>Welcome</h1>
          <p className="text-slate-600 dark:text-slate-400">Sign in or create your account</p>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl border border-slate-200 dark:border-slate-700 p-8">
          <Tabs defaultValue="login" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger data-testid="login-tab" value="login">Sign In</TabsTrigger>
              <TabsTrigger data-testid="signup-tab" value="signup">Sign Up</TabsTrigger>
            </TabsList>

            <TabsContent value="login">
              <form onSubmit={handleLogin} className="space-y-4">
                <div>
                  <Label htmlFor="login-email">Email</Label>
                  <div className="relative mt-1.5">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <Input
                      data-testid="login-email-input"
                      id="login-email"
                      type="email"
                      placeholder="you@example.com"
                      value={loginData.email}
                      onChange={(e) => setLoginData({ ...loginData, email: e.target.value })}
                      className="pl-10 h-12"
                      required
                    />
                  </div>
                </div>
                <div>
                  <Label htmlFor="login-password">Password</Label>
                  <div className="relative mt-1.5">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <Input
                      data-testid="login-password-input"
                      id="login-password"
                      type="password"
                      placeholder="••••••••"
                      value={loginData.password}
                      onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                      className="pl-10 h-12"
                      required
                    />
                  </div>
                </div>
                <Button
                  data-testid="login-submit-button"
                  type="submit"
                  className="w-full h-12 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white"
                  disabled={loading}
                >
                  {loading ? 'Signing in...' : 'Sign In'}
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="signup">
              <form onSubmit={handleSignup} className="space-y-4">
                <div>
                  <Label htmlFor="signup-name">Full Name</Label>
                  <div className="relative mt-1.5">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <Input
                      data-testid="signup-name-input"
                      id="signup-name"
                      type="text"
                      placeholder="John Doe"
                      value={signupData.name}
                      onChange={(e) => setSignupData({ ...signupData, name: e.target.value })}
                      className="pl-10 h-12"
                      required
                    />
                  </div>
                </div>
                <div>
                  <Label htmlFor="signup-email">Email</Label>
                  <div className="relative mt-1.5">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <Input
                      data-testid="signup-email-input"
                      id="signup-email"
                      type="email"
                      placeholder="you@example.com"
                      value={signupData.email}
                      onChange={(e) => setSignupData({ ...signupData, email: e.target.value })}
                      className="pl-10 h-12"
                      required
                    />
                  </div>
                </div>
                <div>
                  <Label htmlFor="signup-password">Password</Label>
                  <div className="relative mt-1.5">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <Input
                      data-testid="signup-password-input"
                      id="signup-password"
                      type="password"
                      placeholder="••••••••"
                      value={signupData.password}
                      onChange={(e) => setSignupData({ ...signupData, password: e.target.value })}
                      className="pl-10 h-12"
                      required
                    />
                  </div>
                </div>
                <div>
                  <Label htmlFor="signup-role">I am a</Label>
                  <div className="relative mt-1.5">
                    <UserCircle className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <select
                      data-testid="signup-role-select"
                      id="signup-role"
                      value={signupData.role}
                      onChange={(e) => setSignupData({ ...signupData, role: e.target.value })}
                      className="w-full h-12 pl-10 pr-4 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
                    >
                      <option value="organizer">Event Organizer</option>
                      <option value="attendee">Event Attendee</option>
                    </select>
                  </div>
                </div>
                <Button
                  data-testid="signup-submit-button"
                  type="submit"
                  className="w-full h-12 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white"
                  disabled={loading}
                >
                  {loading ? 'Creating account...' : 'Create Account'}
                </Button>
              </form>
            </TabsContent>
          </Tabs>
        </div>

        <div className="mt-6 text-center">
          <Button
            data-testid="back-home-button"
            variant="ghost"
            onClick={() => navigate('/')}
            className="text-slate-600 dark:text-slate-400"
          >
            Back to Home
          </Button>
        </div>
      </motion.div>
    </div>
  );
};

export default Auth;
