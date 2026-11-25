"use client";
import { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";

interface AuthImageProps {
    url: string;
    alt: string;
    token: string | null;
}

/**
 * Fetches and renders an image that requires Bearer authentication.
 * Uses a blob URL to avoid exposing the token in the <img src>.
 */
export default function AuthImage({ url, alt, token }: AuthImageProps) {
    const [blobUrl, setBlobUrl] = useState<string | null>(null);
    const [error, setError] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!token) return;
        let cancelled = false;

        fetch(url, { headers: { Authorization: `Bearer ${token}` } })
            .then((res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.blob();
            })
            .then((blob) => {
                if (!cancelled) {
                    setBlobUrl(URL.createObjectURL(blob));
                    setLoading(false);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setError(true);
                    setLoading(false);
                }
            });

        return () => {
            cancelled = true;
            if (blobUrl) URL.revokeObjectURL(blobUrl);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [url, token]);

    if (loading) return <Loader2 className="w-8 h-8 animate-spin text-neutral-500" />;
    if (error || !blobUrl)
        return <p className="text-neutral-600 text-sm italic">Not available</p>;

    return <img src={blobUrl} alt={alt} className="max-w-full max-h-full object-contain" />;
}
