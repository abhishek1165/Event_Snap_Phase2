import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, ArrowLeft, Eye, EyeOff } from "lucide-react";
import { toast } from "sonner";
import api from "@/utils/api";
import { cn } from "@/lib/utils";
import Logo from "@/components/Logo";
import { Button, Input, Label } from "@/components/brand/atoms";
import { T, R, EASE } from "@/design/tokens";

const Auth = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("login");
  const [showPw, setShowPw] = useState(false);
  const [loginData, setLoginData] = useState({ email: "", password: "" });
  const [signupData, setSignupData] = useState({
    email: "",
    password: "",
    name: "",
    role: "organizer",
  });

  // ── API contracts preserved exactly (utils/api.js, localStorage keys, payloads) ──
  const finishAuth = (response) => {
    localStorage.setItem("token", response.data.token);
    localStorage.setItem("user", JSON.stringify(response.data.user));
    if (response.data.user.role === "organizer") navigate("/dashboard");
    else navigate("/attendjoin");
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await api.post("/auth/login", loginData);
      toast.success("Welcome back.");
      finishAuth(response);
    } catch (error) {
      toast.error(error.response?.data?.detail || "Login failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await api.post("/auth/register", signupData);
      toast.success("Account created.");
      finishAuth(response);
    } catch (error) {
      toast.error(error.response?.data?.detail || "Signup failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-[100dvh] bg-ink text-text">
      <Ambient />

      {/* top-left back link */}
      <button
        onClick={() => navigate("/")}
        className="absolute left-6 top-6 z-20 inline-flex items-center gap-2 text-sm font-medium text-text-dim transition-colors hover:text-text sm:left-8 sm:top-8"
      >
        <ArrowLeft className="h-4 w-4 transition-transform group-hover:-translate-x-1" />
        Back
      </button>

      <div className="mx-auto grid min-h-[100dvh] max-w-6xl grid-cols-1 lg:grid-cols-2">
        {/* ── LEFT: brand / aperture visual (hidden on mobile) ── */}
        <div className="relative hidden flex-col justify-between overflow-hidden border-r border-frame-soft p-12 lg:flex">
          <Logo size={30} variant="full" onClick={() => navigate("/")} />
          <div className="relative max-w-md">
            <h1 className="text-4xl font-semibold leading-[1.08] tracking-tight" style={R}>
              One selfie in.
              <br />
              <span style={{ background: T.heroText, WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>
                Every photo of you, out.
              </span>
            </h1>
            <p className="mt-5 text-text-dim">
              Face-matched event galleries. Attendees find themselves in seconds, organizers stop answering "send me that photo".
            </p>
            {/* mini aperture marks as a quiet decorative strip (CSS, not fake screenshot) */}
            <div className="mt-10 flex gap-3 opacity-60">
              {[0, 1, 2, 3].map((i) => (
                <span key={i} className="size-9 rounded-lg border border-iris/40" style={{ position: "relative" }}>
                  <span className="absolute inset-2 rounded-full border border-iris/30" />
                  <span className="absolute inset-[14px] rounded-full bg-match/70" />
                </span>
              ))}
            </div>
          </div>
          <p className="font-mono text-[0.7rem] uppercase tracking-[0.14em] text-text-faint">
            Face matching for event photography
          </p>
        </div>

        {/* ── RIGHT: form ── */}
        <div className="flex items-center justify-center p-6 sm:p-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: EASE.out }}
            className="w-full max-w-md"
          >
            {/* mobile brand */}
            <div className="mb-10 lg:hidden">
              <Logo size={28} onClick={() => navigate("/")} />
            </div>

            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl" style={R}>
              {activeTab === "login" ? "Sign in" : "Create account"}
            </h2>
            <p className="mt-2 text-text-dim">
              {activeTab === "login" ? "Welcome back. Enter your details to continue." : "Start managing your event galleries."}
            </p>

            {/* tabs */}
            <div className="mt-8 flex items-center gap-8 border-b border-frame-soft pb-2">
              {["login", "signup"].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={cn(
                    "relative pb-2 text-sm font-medium tracking-wide transition-colors",
                    activeTab === tab ? "text-text" : "text-text-faint hover:text-text-dim"
                  )}
                >
                  {tab === "login" ? "Log in" : "Sign up"}
                  {activeTab === tab && (
                    <motion.div layoutId="auth-tab" className="absolute -bottom-[9px] left-0 right-0 h-[2px] bg-iris-2" />
                  )}
                </button>
              ))}
            </div>

            {/* forms */}
            <div className="relative mt-8">
              <AnimatePresence mode="wait">
                {activeTab === "login" ? (
                  <motion.form
                    key="login"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    transition={{ duration: 0.25, ease: EASE.out }}
                    onSubmit={handleLogin}
                    className="space-y-5"
                  >
                    <div>
                      <Label htmlFor="login-email">Email</Label>
                      <Input
                        id="login-email"
                        type="email"
                        autoComplete="email"
                        required
                        placeholder="you@example.com"
                        value={loginData.email}
                        onChange={(e) => setLoginData({ ...loginData, email: e.target.value })}
                      />
                    </div>
                    <div>
                      <Label htmlFor="login-password">Password</Label>
                      <PasswordInput
                        id="login-password"
                        autoComplete="current-password"
                        value={loginData.password}
                        onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                        show={showPw}
                        onToggle={() => setShowPw((s) => !s)}
                      />
                    </div>
                    <Button type="submit" size="lg" className="w-full" disabled={loading}>
                      {loading ? "Signing in…" : (<>Continue <ArrowRight className="h-4 w-4" /></>)}
                    </Button>
                  </motion.form>
                ) : (
                  <motion.form
                    key="signup"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    transition={{ duration: 0.25, ease: EASE.out }}
                    onSubmit={handleSignup}
                    className="space-y-5"
                  >
                    <div>
                      <Label htmlFor="su-name">Full name</Label>
                      <Input
                        id="su-name"
                        type="text"
                        autoComplete="name"
                        required
                        placeholder="Jordan Avery"
                        value={signupData.name}
                        onChange={(e) => setSignupData({ ...signupData, name: e.target.value })}
                      />
                    </div>
                    <div>
                      <Label htmlFor="su-email">Email</Label>
                      <Input
                        id="su-email"
                        type="email"
                        autoComplete="email"
                        required
                        placeholder="you@example.com"
                        value={signupData.email}
                        onChange={(e) => setSignupData({ ...signupData, email: e.target.value })}
                      />
                    </div>
                    <div>
                      <Label htmlFor="su-password">Password</Label>
                      <PasswordInput
                        id="su-password"
                        autoComplete="new-password"
                        value={signupData.password}
                        onChange={(e) => setSignupData({ ...signupData, password: e.target.value })}
                        show={showPw}
                        onToggle={() => setShowPw((s) => !s)}
                      />
                    </div>

                    {/* role segmented control */}
                    <div>
                      <Label>I am joining as</Label>
                      <div className="grid grid-cols-2 gap-1 rounded-xl border border-frame bg-surface-2 p-1">
                        {[
                          ["organizer", "Organizer"],
                          ["attendee", "Attendee"],
                        ].map(([val, label]) => (
                          <button
                            key={val}
                            type="button"
                            onClick={() => setSignupData({ ...signupData, role: val })}
                            className={cn(
                              "relative rounded-lg py-2.5 text-sm font-semibold transition-colors",
                              signupData.role === val ? "text-text" : "text-text-faint hover:text-text-dim"
                            )}
                          >
                            {signupData.role === val && (
                              <motion.div layoutId="role-pill" className="absolute inset-0 rounded-lg bg-surface-3" transition={EASE.spring} />
                            )}
                            <span className="relative">{label}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    <Button type="submit" variant="gradient" size="lg" className="w-full" disabled={loading}>
                      {loading ? "Creating…" : (<>Create account <ArrowRight className="h-4 w-4" /></>)}
                    </Button>
                  </motion.form>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

/* password input with show/hide toggle (emil/impeccable form patterns) */
function PasswordInput({ id, value, onChange, show, onToggle, autoComplete, required }) {
  return (
    <div className="relative">
      <Input
        id={id}
        type={show ? "text" : "password"}
        autoComplete={autoComplete}
        required={required}
        placeholder="••••••••"
        value={value}
        onChange={onChange}
        className="pr-12"
      />
      <button
        type="button"
        onClick={onToggle}
        aria-label={show ? "Hide password" : "Show password"}
        className="absolute right-3 top-3.5 flex h-6 w-6 items-center justify-center rounded-md text-text-faint transition-colors hover:text-text"
      >
        {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
      </button>
    </div>
  );
}

function Ambient() {
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 bg-ink" />
      <div className="absolute -top-40 left-1/4 h-[480px] w-[480px] rounded-full bg-iris/[0.08] blur-[120px]" />
      <div className="absolute bottom-0 right-0 h-[400px] w-[400px] rounded-full bg-match/[0.05] blur-[120px]" />
    </div>
  );
}

export default Auth;
