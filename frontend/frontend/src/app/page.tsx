"use client";

import { useAuth } from "@/components/AuthContext";
import DashboardMVP from "@/components/dashboard/DashboardMVP";
import Login from "@/components/auth/Login";
import { useEffect, useState } from "react";

export default function Home() {
    const { token, isLoading } = useAuth();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted || isLoading) return <div className="p-8">Loading...</div>;

    return (
        <main>
            {token ? <DashboardMVP /> : <Login />}
        </main>
    );
}
