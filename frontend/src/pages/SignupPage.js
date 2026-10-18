import React, { useMemo, useState } from "react";
import { ArrowRight, CheckCircle2, Eye, EyeOff } from "lucide-react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import AuthShell from "../components/AuthShell";
import api from "@/utils/api";

const planLabels = {
  starter: "Starter — free",
  studio: "Studio — $19 / month",
  enterprise: "Enterprise — tailored",
};

export default function SignupPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const plan = params.get("plan") || "starter";
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    role: "organizer",
  });

  const planLabel = useMemo(() => planLabels[plan] ?? planLabels.starter, [plan]);

  const update = (event) => {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
    if (error) setError("");
  };

  const submit = async (event) => {
    event.preventDefault();
    if (!form.name.trim() || !form.email.trim() || !form.password || !form.confirmPassword) {
      return setError("Please complete every field.");
    }
    if (!/^\S+@\S+\.\S+$/.test(form.email)) {
      return setError("Enter a valid email address.");
    }
    if (form.password.length < 6) {
      return setError("Choose a password with at least 6 characters.");
    }
    if (form.password !== form.confirmPassword) {
      return setError("Your passwords don't match.");
    }

    setLoading(true);
    setError("");

    try {
      const payload = {
        name: form.name.trim(),
        email: form.email.trim(),
        password: form.password,
        role: form.role,
      };

      const response = await api.post("/auth/register", payload);
      localStorage.setItem("token", response.data.token);
      localStorage.setItem("user", JSON.stringify(response.data.user));
      toast.success("Account created successfully!");

      if (response.data.user?.role === "organizer") {
        navigate("/dashboard", { replace: true });
      } else {
        navigate("/attendjoin", { replace: true });
      }
    } catch (err) {
      const msg = err.response?.data?.detail || "Signup failed. Please try again.";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell mode="signup">
      <p className="eyebrow">START YOUR HIGHLIGHT REEL</p>
      <h1>Make every guest feel like the main character.</h1>
      <p className="auth-lede">Create your account and let the good photos find their people.</p>
      
      <div className="selected-plan">
        <CheckCircle2 size={17} />
        <span>
          Selected plan: <strong>{planLabel}</strong>
        </span>
      </div>

      <form className="auth-form" onSubmit={submit} noValidate>
        <label>
          Full name
          <input
            name="name"
            value={form.name}
            onChange={update}
            autoComplete="name"
            placeholder="Priya Sharma"
            required
          />
        </label>

        <label>
          Email address
          <input
            name="email"
            type="email"
            value={form.email}
            onChange={update}
            autoComplete="email"
            placeholder="you@example.com"
            required
          />
        </label>

        <label>
          Account type
          <select
            name="role"
            value={form.role}
            onChange={update}
            className="cursor-pointer"
          >
            <option value="organizer">Event Organizer (Host events & galleries)</option>
            <option value="attendee">Attendee (Find & download my photos)</option>
          </select>
        </label>

        <label>
          Password
          <span className="input-wrap">
            <input
              name="password"
              type={showPassword ? "text" : "password"}
              value={form.password}
              onChange={update}
              autoComplete="new-password"
              placeholder="At least 6 characters"
              required
            />
            <button
              type="button"
              className="input-icon"
              onClick={() => setShowPassword((value) => !value)}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
            </button>
          </span>
        </label>

        <label>
          Confirm password
          <input
            name="confirmPassword"
            type={showPassword ? "text" : "password"}
            value={form.confirmPassword}
            onChange={update}
            autoComplete="new-password"
            placeholder="Repeat your password"
            required
          />
        </label>

        {error && <p className="form-error" role="alert">{error}</p>}

        <button className="button button--ink button--wide" type="submit" disabled={loading}>
          {loading ? "Creating account..." : (
            <>
              Create my account <ArrowRight size={18} />
            </>
          )}
        </button>
      </form>

      <p className="auth-switch">
        Already have an account? <Link to="/login">Sign in</Link>
      </p>
    </AuthShell>
  );
}
