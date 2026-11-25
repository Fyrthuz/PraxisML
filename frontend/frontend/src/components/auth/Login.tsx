"use client";
import { useState } from "react";
import { useAuth } from "@/components/AuthContext";
import { config } from "@/lib/config";

export default function Login() {
    const [isLogin, setIsLogin] = useState(true);
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [name, setName] = useState("");
    const [tenantName, setTenantName] = useState("");
    const [error, setError] = useState("");
    const [submitting, setSubmitting] = useState(false);
    const { login } = useAuth();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");
        setSubmitting(true);

        try {
            const endpoint = isLogin ? "/api/v1/auth/login" : "/api/v1/auth/register";
            let body: BodyInit;
            let headers: HeadersInit = {};

            if (isLogin) {
                const params = new URLSearchParams();
                params.append("username", email);
                params.append("password", password);
                body = params;
                headers = { "Content-Type": "application/x-www-form-urlencoded" };
            } else {
                body = JSON.stringify({
                    email,
                    password,
                    full_name: name,
                    tenant_name: tenantName || undefined,
                });
                headers = { "Content-Type": "application/json" };
            }

            const res = await fetch(config.API_BASE_URL + endpoint, {
                method: "POST",
                headers,
                body,
            });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.detail || "Authentication failed");
            }

            if (isLogin) {
                const data = await res.json();
                login(data.access_token);
            } else {
                setIsLogin(true);
                setError("Account created. Sign in with your credentials.");
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setSubmitting(false);
        }
    };

    const isSuccess = error.startsWith("Account created");

    return (
        <div className="min-h-screen bg-[#0b0c0f] text-zinc-100 flex items-center justify-center px-4">
            <div className="w-full max-w-sm bg-[#101217] border border-zinc-800 rounded-lg p-7">
                {/* Brand */}
                <div className="flex items-center gap-2.5 mb-6">
                    <div className="w-6 h-6 rounded bg-orange-500 flex items-center justify-center text-[11px] font-bold text-zinc-950 font-mono">
                        P
                    </div>
                    <span className="text-sm font-semibold tracking-tight text-zinc-100">
                        PraxisML
                    </span>
                    <span className="text-[11px] font-mono text-zinc-500 ml-auto">v2</span>
                </div>

                <h2 className="text-lg font-semibold text-zinc-100">
                    {isLogin ? "Sign in to workspace" : "Create workspace account"}
                </h2>
                <p className="text-xs text-zinc-500 mt-1 mb-5">
                    {isLogin
                        ? "Enter your credentials to access datasets, models and jobs."
                        : "First user in a new organization is assigned the admin role."}
                </p>

                {error && (
                    <div
                        className={`px-3 py-2 mb-4 text-xs rounded-md border ${
                            isSuccess
                                ? "bg-emerald-950/50 border-emerald-800/70 text-emerald-300"
                                : "bg-red-950/50 border-red-900/70 text-red-300"
                        }`}
                    >
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-3.5">
                    {!isLogin && (
                        <>
                            <div>
                                <label className="block mb-1 text-xs font-medium text-zinc-400">
                                    Full name
                                </label>
                                <input
                                    type="text"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    className="w-full px-3 py-2 text-sm bg-[#0b0c0f] border border-zinc-800 rounded-md text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block mb-1 text-xs font-medium text-zinc-400">
                                    Organization
                                </label>
                                <input
                                    type="text"
                                    value={tenantName}
                                    onChange={(e) => setTenantName(e.target.value)}
                                    placeholder="Optional"
                                    className="w-full px-3 py-2 text-sm bg-[#0b0c0f] border border-zinc-800 rounded-md text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
                                />
                            </div>
                        </>
                    )}
                    <div>
                        <label className="block mb-1 text-xs font-medium text-zinc-400">
                            Email
                        </label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="name@company.com"
                            className="w-full px-3 py-2 text-sm bg-[#0b0c0f] border border-zinc-800 rounded-md text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
                            required
                        />
                    </div>
                    <div>
                        <label className="block mb-1 text-xs font-medium text-zinc-400">
                            Password
                        </label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full px-3 py-2 text-sm bg-[#0b0c0f] border border-zinc-800 rounded-md text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
                            required
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={submitting}
                        className="w-full mt-2 px-3 py-2 text-sm font-medium text-zinc-950 bg-zinc-100 rounded-md hover:bg-white transition-colors disabled:opacity-50"
                    >
                        {submitting ? "Please wait..." : isLogin ? "Sign in" : "Create account"}
                    </button>
                </form>

                <div className="mt-5 pt-4 border-t border-zinc-800/80 text-xs text-zinc-500 flex items-center justify-between">
                    <span>{isLogin ? "No account yet?" : "Already registered?"}</span>
                    <button
                        type="button"
                        onClick={() => {
                            setIsLogin(!isLogin);
                            setError("");
                        }}
                        className="text-zinc-200 hover:text-white font-medium"
                    >
                        {isLogin ? "Create account" : "Sign in"}
                    </button>
                </div>
            </div>
        </div>
    );
}
