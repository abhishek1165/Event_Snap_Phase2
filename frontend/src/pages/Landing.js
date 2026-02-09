import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Camera, Upload, Search, Zap, Shield, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Landing = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Upload,
      title: 'Batch Upload',
      description: 'Upload hundreds of photos at once with intelligent processing',
      image: 'https://images.unsplash.com/photo-1769374071311-3935864d4a9a'
    },
    {
      icon: Camera,
      title: 'Selfie Search',
      description: 'Take a selfie and instantly find all your photos from the event',
      image: 'https://images.unsplash.com/photo-1758275557389-a8137c03fefc'
    },
    {
      icon: Zap,
      title: 'Lightning Fast',
      description: 'AI-powered face recognition delivers results in seconds',
      image: 'https://images.unsplash.com/photo-1758274251643-66fed4dc51c0'
    }
  ];

  const benefits = [
    { icon: Shield, text: 'Privacy-First: Event-scoped access only' },
    { icon: Clock, text: 'Save Hours: No manual photo searching' },
    { icon: Search, text: 'Find Every Photo: Never miss a moment' }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Floating Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="sticky top-4 z-50 mx-auto max-w-5xl px-4"
      >
        <div className="glass rounded-full border border-slate-200/50 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Camera className="w-6 h-6 text-indigo-600" />
            <span className="font-bold text-xl" style={{ fontFamily: 'Outfit, sans-serif' }}>FaceShot</span>
          </div>
          <div className="flex gap-3">
            <Button
              data-testid="header-login-button"
              variant="ghost"
              onClick={() => navigate('/auth')}
              className="rounded-full"
            >
              Sign In
            </Button>
            <Button
              data-testid="header-get-started-button"
              onClick={() => navigate('/auth')}
              className="rounded-full bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-500/20"
            >
              Get Started
            </Button>
          </div>
        </div>
      </motion.header>

      {/* Hero Section */}
      <section className="relative overflow-hidden px-4 pt-20 pb-32">
        <div className="mx-auto max-w-7xl">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <h1
                className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight mb-6"
                style={{ fontFamily: 'Outfit, sans-serif' }}
              >
                Find Your Face in
                <span className="block text-indigo-600">Thousands of Photos</span>
              </h1>
              <p className="text-lg text-slate-600 dark:text-slate-400 mb-8 max-w-xl">
                Stop searching manually through event photos. Upload a selfie and let AI instantly find every photo you're in.
              </p>
              <div className="flex flex-wrap gap-4">
                <Button
                  data-testid="hero-organizer-button"
                  onClick={() => navigate('/auth')}
                  size="lg"
                  className="rounded-full bg-indigo-600 hover:bg-indigo-700 text-white px-8 shadow-lg shadow-indigo-500/20 hover:-translate-y-0.5 transition-transform"
                >
                  For Organizers
                </Button>
                <Button
                  data-testid="hero-attendee-button"
                  onClick={() => navigate('/attend')}
                  size="lg"
                  variant="outline"
                  className="rounded-full border-slate-300 hover:border-indigo-500 hover:text-indigo-600 transition-colors"
                >
                  Find My Photos
                </Button>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
              className="relative"
            >
              <div className="relative rounded-3xl overflow-hidden border border-slate-200/50 shadow-2xl">
                <img
                  src="https://images.unsplash.com/photo-1560439514-4e9645039924?w=800&h=600&fit=crop"
                  alt="Event crowd"
                  className="w-full h-[500px] object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <div className="absolute bottom-6 left-6 right-6 text-white">
                  <p className="text-sm font-medium mb-1">Wedding • June 2026</p>
                  <p className="text-2xl font-semibold" style={{ fontFamily: 'Outfit, sans-serif' }}>247 Photos Found</p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 bg-slate-50 dark:bg-slate-900">
        <div className="mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold mb-4" style={{ fontFamily: 'Outfit, sans-serif' }}>
              How It Works
            </h2>
            <p className="text-lg text-slate-600 dark:text-slate-400">Simple, fast, and magical</p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ y: -4 }}
                className="group relative"
              >
                <div className="relative h-64 rounded-2xl overflow-hidden mb-6">
                  <img
                    src={feature.image}
                    alt={feature.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
                  <div className="absolute bottom-4 left-4">
                    <feature.icon className="w-10 h-10 text-white" />
                  </div>
                </div>
                <h3 className="text-xl font-semibold mb-2" style={{ fontFamily: 'Outfit, sans-serif' }}>
                  {feature.title}
                </h3>
                <p className="text-slate-600 dark:text-slate-400">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20 px-4">
        <div className="mx-auto max-w-5xl">
          <div className="grid md:grid-cols-3 gap-6">
            {benefits.map((benefit, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center gap-3 p-6 rounded-xl border border-slate-200 dark:border-slate-800 hover:border-indigo-500/50 transition-colors"
              >
                <benefit.icon className="w-6 h-6 text-indigo-600 flex-shrink-0" />
                <p className="font-medium">{benefit.text}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-indigo-600 text-white">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mx-auto max-w-4xl text-center"
        >
          <h2 className="text-3xl sm:text-4xl font-bold mb-6" style={{ fontFamily: 'Outfit, sans-serif' }}>
            Ready to Transform Your Event Photos?
          </h2>
          <p className="text-lg text-indigo-100 mb-8 max-w-2xl mx-auto">
            Join thousands of event organizers using AI-powered face recognition
          </p>
          <Button
            data-testid="cta-get-started-button"
            onClick={() => navigate('/auth')}
            size="lg"
            className="rounded-full bg-white text-indigo-600 hover:bg-slate-50 px-8 shadow-lg hover:-translate-y-0.5 transition-transform"
          >
            Get Started Free
          </Button>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-slate-200 dark:border-slate-800">
        <div className="mx-auto max-w-7xl text-center text-slate-600 dark:text-slate-400">
          <p>&copy; 2026 FaceShot. Privacy-first face recognition platform.</p>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
