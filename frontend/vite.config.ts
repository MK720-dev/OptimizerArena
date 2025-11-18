import { defineConfig, loadEnv } from 'vite';
import plugin from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
    // Load env file based on `mode` in the current working directory.
    // Set the third parameter to '' to load all env regardless of the `VITE_` prefix.
    const env = loadEnv(mode, process.cwd(), '');
    
    const API_URL = env.VITE_API_URL || "http://localhost:8000";
    
    return {
        plugins: [plugin()],
        define: {
            global: 'window',
        },
        server: {
            port: 5173,
            proxy: {
                '/api': {
                    target: API_URL,
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/api/, ''),
                }
            }
        }
    };
});
