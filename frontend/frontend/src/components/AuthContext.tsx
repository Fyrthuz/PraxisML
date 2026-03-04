"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

// The shape of our Auth state
interface AuthState {
    token: string | null;
    user: any | null;
    tenant: any | null;
    isLoading: boolean;
    login: (token: string) => void;
    logout: () => void;
    setTenant: (tenant: any) => void;
}

const AuthContext = createContext<AuthState>({
    token: null,
    user: null,
    tenant: null,
    isLoading: true,
    login: () => { },
    logout: () => { },
    setTenant: () => { },
});

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [token, setToken] = useState<string | null>(null);
    const [user, setUser] = useState<any | null>(null);
    const [tenant, setTenantState] = useState<any | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    // Initialize from LocalStorage
    useEffect(() => {
        const storedToken = localStorage.getItem("token");
        if (storedToken) {
            setToken(storedToken);
            fetchProfile(storedToken);
        } else {
            setIsLoading(false);
        }
    }, []);

    const fetchProfile = async (currentToken: string) => {
        try {
            // Fetch /auth/me for User
            const userRes = await fetch("http://localhost:8000/api/v1/auth/me", {
                headers: { Authorization: `Bearer ${currentToken}` },
            });
            if (userRes.ok) {
                setUser(await userRes.json());
            } else {
                throw new Error("Invalid token");
            }

            // Fetch /tenants/me for Tenant
            const tenantRes = await fetch("http://localhost:8000/api/v1/tenants/me", {
                headers: { Authorization: `Bearer ${currentToken}` },
            });
            if (tenantRes.ok) {
                setTenantState(await tenantRes.json());
            }
        } catch (error) {
            console.error("Auth error:", error);
            logout();
        } finally {
            setIsLoading(false);
        }
    };

    const login = (newToken: string) => {
        localStorage.setItem("token", newToken);
        setToken(newToken);
        setIsLoading(true);
        fetchProfile(newToken);
    };

    const logout = () => {
        localStorage.removeItem("token");
        setToken(null);
        setUser(null);
        setTenantState(null);
        setIsLoading(false);
    };

    const setTenant = (newTenant: any) => {
        setTenantState(newTenant);
    }

    return (
        <AuthContext.Provider value={{ token, user, tenant, isLoading, login, logout, setTenant }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
