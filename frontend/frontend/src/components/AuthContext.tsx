"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { config } from '@/lib/config';

// The shape of our Auth state
interface AuthState {
    token: string | null;
    user: any | null;
    tenant: any | null;
    isLoading: boolean;
    userRole: string;
    login: (token: string) => void;
    logout: () => void;
    setTenant: (tenant: any) => void;
}

const AuthContext = createContext<AuthState>({
    token: null,
    user: null,
    tenant: null,
    isLoading: true,
    userRole: "viewer",
    login: () => { },
    logout: () => { },
    setTenant: () => { },
});

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [token, setToken] = useState<string | null>(null);
    const [user, setUser] = useState<any | null>(null);
    const [tenant, setTenantState] = useState<any | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [userRole, setUserRole] = useState<string>("viewer");

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
            const userRes = await fetch(config.getFullApiUrl("/auth/me"), {
                headers: { Authorization: `Bearer ${currentToken}` },
            });
            if (userRes.ok) {
                const userData = await userRes.json();
                setUser(userData);
                setUserRole(userData.role || "viewer");
            } else {
                throw new Error("Invalid token");
            }

            // Fetch /tenants/me for Tenant
            const tenantRes = await fetch(config.getFullApiUrl("/tenants/me"), {
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
        setUserRole("viewer");
        setIsLoading(false);
    };

    const setTenant = (newTenant: any) => {
        setTenantState(newTenant);
    }

    return (
        <AuthContext.Provider value={{ token, user, tenant, isLoading, userRole, login, logout, setTenant }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
