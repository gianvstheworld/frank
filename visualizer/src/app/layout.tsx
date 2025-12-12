import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "LeRobot Dataset Visualizer",
  description: "Visualization of LeRobot Datasets",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-gray-200 min-h-screen`}>{children}</body>
    </html>
  );
}
