import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { X, Upload, Loader2, FileUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface UploadModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    description: string;
    onUpload: (formData: FormData) => Promise<void>;
    fileAccept?: string;
    requireFile?: boolean;
    hideCloseButton?: boolean;
    fields?: {
        name: string;
        label: string;
        type: 'text' | 'textarea' | 'number' | 'select';
        placeholder?: string;
        options?: { label: string; value: string | number }[];
        required?: boolean;
        defaultValue?: string | number;
    }[];
}

export default function UploadModal({
    isOpen,
    onClose,
    title,
    description,
    onUpload,
    fileAccept,
    requireFile = true,
    hideCloseButton = false,
    fields = []
}: UploadModalProps) {
    const [file, setFile] = useState<File | null>(null);
    const [formValues, setFormValues] = useState<Record<string, any>>(
        fields.reduce((acc, field) => ({ ...acc, [field.name]: field.defaultValue || '' }), {})
    );
    const [isUploading, setIsUploading] = useState(false);

    if (!isOpen) return null;

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (requireFile && !file) return;

        setIsUploading(true);
        try {
            const formData = new FormData();
            if (file) {
                formData.append('file', file);
            }
            Object.entries(formValues).forEach(([key, value]) => {
                formData.append(key, value.toString());
            });
            await onUpload(formData);
            if (!hideCloseButton) {
                onClose();
            }
            setFile(null);
            setFormValues(fields.reduce((acc, field) => ({ ...acc, [field.name]: field.defaultValue || '' }), {}));
        } catch (error) {
            console.error("Upload/Submit failed", error);
            alert("Action failed. Please try again.");
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-neutral-900 border border-neutral-800 w-full max-w-lg rounded-3xl overflow-hidden shadow-2xl scale-in-center">
                <div className="p-6 border-b border-neutral-800 flex justify-between items-center bg-neutral-900/50">
                    <div>
                        <h3 className="text-xl font-bold text-white">{title}</h3>
                        <p className="text-sm text-neutral-400 mt-1">{description}</p>
                    </div>
                    {!hideCloseButton && (
                        <button onClick={onClose} className="p-2 hover:bg-neutral-800 rounded-full text-neutral-400 transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    )}
                </div>

                <form onSubmit={handleSubmit} className="p-8 space-y-6">
                    {/* File Drop Area */}
                    {requireFile && (
                        <div
                            className={cn(
                                "relative group border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-200",
                                file ? "border-indigo-500 bg-indigo-500/5" : "border-neutral-700 hover:border-neutral-600 hover:bg-neutral-800/50"
                            )}
                        >
                            <input
                                type="file"
                                accept={fileAccept}
                                onChange={handleFileChange}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                            />
                            <div className="space-y-3">
                                <div className={cn(
                                    "mx-auto w-12 h-12 rounded-xl flex items-center justify-center transition-colors",
                                    file ? "bg-indigo-500 text-white" : "bg-neutral-800 text-neutral-400 group-hover:text-neutral-300"
                                )}>
                                    {file ? <FileUp className="w-6 h-6" /> : <Upload className="w-6 h-6" />}
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-white">
                                        {file ? file.name : "Click or drag to upload"}
                                    </p>
                                    <p className="text-xs text-neutral-500 mt-1">
                                        {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "Support for various formats"}
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Dynamic Fields */}
                    <div className="space-y-4">
                        {fields.map(field => (
                            <div key={field.name} className="space-y-1.5">
                                <label className="text-xs font-bold text-neutral-500 uppercase ml-1">
                                    {field.label} {field.required && <span className="text-red-500">*</span>}
                                </label>
                                {field.type === 'textarea' ? (
                                    <textarea
                                        required={field.required}
                                        placeholder={field.placeholder}
                                        value={formValues[field.name]}
                                        onChange={e => setFormValues({ ...formValues, [field.name]: e.target.value })}
                                        className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 min-h-[100px] resize-none"
                                    />
                                ) : field.type === 'select' ? (
                                    <select
                                        required={field.required}
                                        value={formValues[field.name]}
                                        onChange={e => setFormValues({ ...formValues, [field.name]: e.target.value })}
                                        className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none"
                                    >
                                        {field.options?.map(opt => (
                                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                                        ))}
                                    </select>
                                ) : (
                                    <input
                                        type={field.type}
                                        required={field.required}
                                        placeholder={field.placeholder}
                                        value={formValues[field.name]}
                                        onChange={e => setFormValues({ ...formValues, [field.name]: e.target.value })}
                                        className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                                    />
                                )}
                            </div>
                        ))}
                    </div>

                    <div className="pt-4 flex gap-3">
                        {!hideCloseButton && (
                            <Button type="button" variant="ghost" className="flex-1 rounded-xl" onClick={onClose} disabled={isUploading}>
                                Cancel
                            </Button>
                        )}
                        <Button type="submit" className="flex-[2] bg-indigo-600 hover:bg-indigo-700 rounded-xl shadow-lg shadow-indigo-600/20" disabled={(requireFile && !file) || isUploading}>
                            {isUploading ? (
                                <>
                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                    Uploading...
                                </>
                            ) : (
                                "Start Upload"
                            )}
                        </Button>
                    </div>
                </form>
            </div>
        </div>
    );
}
