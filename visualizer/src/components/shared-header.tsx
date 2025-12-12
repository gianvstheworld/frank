"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function SharedHeader() {
    const pathname = usePathname();

    // Determine which tab is active
    const isComparison = pathname === "/comparison" || pathname === "/comparison/";
    const isEpisode = pathname.includes("/local/frank/episode_");
    const isHome = pathname === "/" || pathname === "";

    return (
        <header className="border-b border-slate-700/50 backdrop-blur-sm bg-slate-900/80 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
                    <span className="text-2xl">🤖</span>
                    <div>
                        <h1 className="text-xl font-bold text-white">FRANK Robot</h1>
                        <p className="text-xs text-slate-400">Performance Analysis</p>
                    </div>
                </Link>
                <nav className="flex items-center gap-4">
                    <Link
                        href="/"
                        className={`transition-colors ${isHome && !isEpisode ? "text-sky-400 font-semibold" : "text-slate-400 hover:text-white"}`}
                    >
                        Episodes
                    </Link>
                    <Link
                        href="/comparison"
                        className={`transition-colors ${isComparison ? "text-sky-400 font-semibold" : "text-slate-400 hover:text-white"}`}
                    >
                        Comparison
                    </Link>
                </nav>
            </div>
        </header>
    );
}
