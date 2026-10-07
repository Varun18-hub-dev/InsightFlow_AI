import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User } from '@/types';

interface AuthStore {
  token: string | null;
  user: User | null;
  isAuthenticated: boolean;
  setToken: (token: string) => void;
  setUser: (user: User) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set) => ({
      token: null,
      user: null,
      isAuthenticated: false,
      setToken: (token) => set({ token, isAuthenticated: true }),
      setUser: (user) => set({ user }),
      logout: () => set({ token: null, user: null, isAuthenticated: false })
    }),
    { name: 'auth-storage' }
  )
);

interface ChatStore {
  activeConversationId: string | null;
  setActiveConversationId: (id: string | null) => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      activeConversationId: null,
      setActiveConversationId: (id) => set({ activeConversationId: id })
    }),
    { name: 'insightflow-chat-storage' }
  )
);