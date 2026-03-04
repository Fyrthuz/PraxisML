import { useState } from "react";
import { useAuth } from "@/components/AuthContext";

export default function Login() {
    const [isLogin, setIsLogin] = useState(true);
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [name, setName] = useState("");
    const [tenantName, setTenantName] = useState("");
    const [error, setError] = useState("");
    const { login } = useAuth();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");

        try {
            const endpoint = isLogin ? "/api/v1/auth/login" : "/api/v1/auth/register";

            let body;
            let headers: HeadersInit = {};

            if (isLogin) {
                // OAuth2 Password Request Form format
                body = new URLSearchParams();
                body.append("username", email);
                body.append("password", password);
                headers = { "Content-Type": "application/x-www-form-urlencoded" };
            } else {
                // JSON for registration
                body = JSON.stringify({ email, password, full_name: name, tenant_name: tenantName || undefined });
                headers = { "Content-Type": "application/json" };
            }

            const res = await fetch(`http://localhost:8000${endpoint}`, {
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
                // Auto-login after registration could go here, for now switch to login
                setIsLogin(true);
                setError("Registration successful! Please login.");
            }
        } catch (err: any) {
            setError(err.message);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100">
            <div className="w-full max-w-md p-8 bg-white rounded-lg shadow-md">
                <h2 className="mb-6 text-2xl font-bold text-center text-gray-800">
                    {isLogin ? "Sign In" : "Create Account"}
                </h2>

                {error && (
                    <div className={`p-3 mb-4 text-sm rounded ${error.includes('successful') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    {!isLogin && (
                        <>
                            <div>
                                <label className="block mb-1 text-sm font-medium text-gray-700">Full Name</label>
                                <input
                                    type="text"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    className="w-full px-4 py-2 border rounded-md text-black focus:ring-blue-500 focus:border-blue-500"
                                    required={!isLogin}
                                />
                            </div>
                            <div>
                                <label className="block mb-1 text-sm font-medium text-gray-700">Tenant / Organization Name</label>
                                <input
                                    type="text"
                                    value={tenantName}
                                    onChange={(e) => setTenantName(e.target.value)}
                                    placeholder="Optional"
                                    className="w-full px-4 py-2 border rounded-md text-black focus:ring-blue-500 focus:border-blue-500"
                                />
                            </div>
                        </>
                    )}
                    <div>
                        <label className="block mb-1 text-sm font-medium text-gray-700">Email</label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full px-4 py-2 border rounded-md text-black focus:ring-blue-500 focus:border-blue-500"
                            required
                        />
                    </div>
                    <div>
                        <label className="block mb-1 text-sm font-medium text-gray-700">Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full px-4 py-2 border rounded-md text-black focus:ring-blue-500 focus:border-blue-500"
                            required
                        />
                    </div>

                    <button
                        type="submit"
                        className="w-full px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        {isLogin ? "Sign In" : "Sign Up"}
                    </button>
                </form>

                <p className="mt-4 text-sm text-center text-gray-600">
                    {isLogin ? "Don't have an account? " : "Already have an account? "}
                    <button
                        onClick={() => {
                            setIsLogin(!isLogin);
                            setError("");
                        }}
                        className="text-blue-600 hover:underline hover:text-blue-800"
                    >
                        {isLogin ? "Sign Up" : "Sign In"}
                    </button>
                </p>
            </div>
        </div>
    );
}
