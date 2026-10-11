/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
  readonly VITE_API_PORT?: string
  readonly VITE_ELECTRON?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare const __API_BASE_URL__: string;
declare const __ELECTRON_MODE__: boolean;
