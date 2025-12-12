import type { NextConfig } from "next";
import packageJson from './package.json';

const basePath = process.env.NODE_ENV === 'production' ? '/frank' : '';

const nextConfig: NextConfig = {
  // Enable static export for GitHub Pages
  output: 'export',

  // Base path for GitHub Pages (repo name)
  basePath: basePath,

  // Trailing slash for static hosting
  trailingSlash: true,

  // Images need to be unoptimized for static export
  images: {
    unoptimized: true,
  },

  // Expose basePath to client-side code
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },

  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  generateBuildId: () => packageJson.version,
};

export default nextConfig;
