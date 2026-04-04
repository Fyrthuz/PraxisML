"use client";

import { useState, useEffect } from "react";
import { Loader2, UserCog, Shield, Users, AlertCircle, Plus, X, Mail } from "lucide-react";
import toast from "react-hot-toast";
import { config } from '@/lib/config';

interface User {
  id: string;
  email: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
  tenant_id: string;
  created_at: string;
}

interface UsersTabProps {
  token: string | null;
  currentUserRole: string;
  tenantId: string;
  onRefreshUsers?: () => void;
}

const ROLE_COLORS = {
  admin: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  editor: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  viewer: "bg-neutral-700 text-neutral-400 border-neutral-600",
};

const ROLE_LABELS = {
  admin: "Admin",
  editor: "Editor",
  viewer: "Viewer",
};

export default function UsersTab({ token, currentUserRole, tenantId, onRefreshUsers }: UsersTabProps) {
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUpdating, setIsUpdating] = useState<string | null>(null);
  const [selectedRoleChange, setSelectedRoleChange] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [newUser, setNewUser] = useState({ email: "", password: "", full_name: "", role: "viewer" });

  useEffect(() => {
    fetchUsers();
  }, [token, tenantId]);

  const fetchUsers = async () => {
    if (!token || !tenantId) return;

    setIsLoading(true);
    try {
      const res = await fetch(config.getFullApiUrl("/users/"), {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setUsers(data);
      } else {
        toast.error("Failed to load users");
      }
    } catch (err) {
      console.error("Error loading users:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRoleChange = async (userId: string, newRole: string) => {
    if (!token) return;

    setIsUpdating(userId);
    try {
      const res = await fetch(config.getFullApiUrl("/users/${userId}/role"), {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ role: newRole }),
      });

      if (res.ok) {
        toast.success("Role updated successfully");
        setUsers(users.map(u => u.id === userId ? { ...u, role: newRole } : u));
        setSelectedRoleChange(null);
        if (onRefreshUsers) onRefreshUsers();
      } else {
        const data = await res.json();
        toast.error(data.detail || "Failed to update role");
      }
    } catch (err) {
      toast.error("Error updating role");
    } finally {
      setIsUpdating(null);
    }
  };

  const handleCreateUser = async () => {
    if (!token || !newUser.email || !newUser.password) {
      toast.error("Email and password are required");
      return;
    }

    setIsCreating(true);
    try {
      const res = await fetch(config.getFullApiUrl("/users/"), {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newUser),
      });

      if (res.ok) {
        toast.success("User created successfully");
        setShowCreateModal(false);
        setNewUser({ email: "", password: "", full_name: "", role: "viewer" });
        fetchUsers();
        if (onRefreshUsers) onRefreshUsers();
      } else {
        const data = await res.json();
        toast.error(data.detail || "Failed to create user");
      }
    } catch (err) {
      toast.error("Error creating user");
    } finally {
      setIsCreating(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
      </div>
    );
  }

  if (currentUserRole !== "admin") {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-neutral-400">
        <Shield className="w-12 h-12 mb-4 opacity-50" />
        <p className="text-lg font-medium">Access Restricted</p>
        <p className="text-sm">Only administrators can manage users.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Users className="w-6 h-6 text-indigo-400" />
          <div>
            <h3 className="text-lg font-semibold">Team Members</h3>
            <p className="text-sm text-neutral-400">Manage user roles and permissions</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchUsers}
            className="px-4 py-2 text-sm bg-neutral-800 hover:bg-neutral-700 rounded-lg transition-colors"
          >
            Refresh
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add User
          </button>
        </div>
      </div>

      {/* Users List */}
      <div className="bg-neutral-900/50 rounded-xl border border-neutral-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-neutral-900 border-b border-neutral-800">
            <tr>
              <th className="text-left px-6 py-4 text-xs font-medium text-neutral-400 uppercase tracking-wider">User</th>
              <th className="text-left px-6 py-4 text-xs font-medium text-neutral-400 uppercase tracking-wider">Role</th>
              <th className="text-left px-6 py-4 text-xs font-medium text-neutral-400 uppercase tracking-wider">Status</th>
              <th className="text-left px-6 py-4 text-xs font-medium text-neutral-400 uppercase tracking-wider">Joined</th>
              <th className="text-right px-6 py-4 text-xs font-medium text-neutral-400 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {users.map((user) => (
              <tr key={user.id} className="hover:bg-neutral-800/50 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-sm font-medium">
                      {user.full_name?.[0]?.toUpperCase() || user.email[0].toUpperCase()}
                    </div>
                    <div>
                      <p className="font-medium text-white">{user.full_name || "No name"}</p>
                      <p className="text-sm text-neutral-400">{user.email}</p>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className={`px-3 py-1 rounded-full text-xs font-medium border ${ROLE_COLORS[user.role as keyof typeof ROLE_COLORS] || ROLE_COLORS.viewer}`}>
                    {ROLE_LABELS[user.role as keyof typeof ROLE_LABELS] || user.role}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className={`text-sm ${user.is_active ? "text-emerald-400" : "text-neutral-500"}`}>
                    {user.is_active ? "Active" : "Inactive"}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-neutral-400">
                  {user.created_at ? new Date(user.created_at).toLocaleDateString() : "-"}
                </td>
                <td className="px-6 py-4 text-right">
                  {selectedRoleChange === user.id ? (
                    <div className="flex items-center gap-2 justify-end">
                      <select
                        className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-1.5 text-sm text-white"
                        defaultValue={user.role}
                        onChange={(e) => handleRoleChange(user.id, e.target.value)}
                        disabled={isUpdating === user.id}
                      >
                        <option value="admin">Admin</option>
                        <option value="editor">Editor</option>
                        <option value="viewer">Viewer</option>
                      </select>
                      <button
                        onClick={() => setSelectedRoleChange(null)}
                        className="px-3 py-1.5 text-sm text-neutral-400 hover:text-white"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setSelectedRoleChange(user.id)}
                      className="px-3 py-1.5 text-sm bg-neutral-800 hover:bg-neutral-700 rounded-lg transition-colors"
                    >
                      Change Role
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {users.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-neutral-400">
            <Users className="w-12 h-12 mb-4 opacity-50" />
            <p>No users found in this organization</p>
          </div>
        )}
      </div>

      {/* Create User Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-neutral-900 rounded-xl border border-neutral-800 p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">Add New User</h3>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-neutral-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-neutral-400 mb-2">Email</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-500" />
                  <input
                    type="email"
                    value={newUser.email}
                    onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                    className="w-full bg-neutral-800 border border-neutral-700 rounded-lg py-2.5 pl-10 pr-4 text-white placeholder-neutral-500 focus:outline-none focus:border-indigo-500"
                    placeholder="user@example.com"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-400 mb-2">Full Name (optional)</label>
                <input
                  type="text"
                  value={newUser.full_name}
                  onChange={(e) => setNewUser({ ...newUser, full_name: e.target.value })}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg py-2.5 px-4 text-white placeholder-neutral-500 focus:outline-none focus:border-indigo-500"
                  placeholder="John Doe"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-400 mb-2">Password</label>
                <input
                  type="password"
                  value={newUser.password}
                  onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg py-2.5 px-4 text-white placeholder-neutral-500 focus:outline-none focus:border-indigo-500"
                  placeholder="••••••••"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-400 mb-2">Role</label>
                <select
                  value={newUser.role}
                  onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg py-2.5 px-4 text-white focus:outline-none focus:border-indigo-500"
                >
                  <option value="viewer">Viewer</option>
                  <option value="editor">Editor</option>
                  <option value="admin">Admin</option>
                </select>
                <p className="text-xs text-neutral-500 mt-2">New users will be created as Viewer by default</p>
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 text-sm text-neutral-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateUser}
                disabled={isCreating || !newUser.email || !newUser.password}
                className="px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
              >
                {isCreating ? <Loader2 className="w-4 h-4 animate-spin" /> : "Create User"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}