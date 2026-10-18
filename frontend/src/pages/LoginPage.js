import React, { useState } from "react";
import { ArrowRight, Eye, EyeOff } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import AuthShell from "../components/AuthShell";
import api from "@/utils/api";

export default function LoginPage() {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({ email: "", password: "" });

  const update = (event) => {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
    if (error) setError("");
  };

  const submit = async (event) => {
    event.preventDefault();
    if (!form.email || !form.password) {
      return setError("Enter your email and password to continue.");
    }

    setLoading(true);
    setError("");

    try {
      const response = await api.post("/auth/login", {
        email: form.email.trim(),
        password: form.password,
      });

      localStorage.setItem("token", response.data.token);
      localStorage.setItem("user", JSON.stringify(response.data.user));
      toast.success("Welcome back!");

      if (response.data.user?.role === "organizer") {
        navigate("/dashboard", { replace: true });
      } else {
        navigate("/attendjoin", { replace: true });
      }
    } catch (err) {
      const msg = err.response?.data?.detail || "Invalid email or password.";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell mode="login">
      <p className="eyebrow">THE GALLERY IS OPEN</p>
      <h1>Welcome back.</h1>
      <p className="auth-lede">Sign in to keep every frame within reach.</p>
      <form className="auth-form auth-form--short" onSubmit={submit} noValidate>
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
          Password
          <span className="input-wrap">
            <input
              name="password"
              type={showPassword ? "text" : "password"}
              value={form.password}
              onChange={update}
              autoComplete="current-password"
              placeholder="Your password"
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
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="button button--ink button--wide" type="submit" disabled={loading}>
          {loading ? "Entering gallery..." : (
            <>
              Enter my gallery <ArrowRight size={18} />
            </>
          )}
        </button>
      </form>
      <p className="auth-switch">
        New here? <Link to="/signup">Create an account</Link>
      </p>
    </AuthShell>
  );
}
