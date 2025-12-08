"use client";

import Link from "next/link";
import React, { useEffect, useState } from "react";
import { parquetReadObjects } from "hyparquet";

interface SidebarProps {
  datasetInfo: any;
  paginatedEpisodes: any[];
  episodeId: any;
  totalPages: number;
  currentPage: number;
  prevPage: () => void;
  nextPage: () => void;
}

interface EpisodeInfo {
  id: number;
  task: string;
}

// Hugging Face configuration
const HF_REPO = "tommaselli/frank_load_experiments";
const HF_BASE = `https://huggingface.co/datasets/${HF_REPO}/resolve/main`;

const Sidebar: React.FC<SidebarProps> = ({
  datasetInfo,
  paginatedEpisodes,
  episodeId,
  totalPages,
  currentPage,
  prevPage,
  nextPage,
}) => {
  const [sidebarVisible, setSidebarVisible] = useState(true);
  const [episodeInfos, setEpisodeInfos] = useState<EpisodeInfo[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);

  const toggleSidebar = () => setSidebarVisible((prev) => !prev);

  const sidebarRef = React.useRef<HTMLDivElement>(null);

  // Load episode metadata (task names) from HF
  useEffect(() => {
    const loadEpisodeMetadata = async () => {
      try {
        const tasksRes = await fetch(`${HF_BASE}/meta/tasks.parquet`);
        const tasksBuffer = await tasksRes.arrayBuffer();
        const tasksData = await parquetReadObjects({ file: tasksBuffer }) as any[];

        const episodesRes = await fetch(`${HF_BASE}/meta/episodes/chunk-000/file-000.parquet`);
        const episodesBuffer = await episodesRes.arrayBuffer();
        const episodesData = await parquetReadObjects({ file: episodesBuffer }) as any[];

        const infos = episodesData.map((ep, idx) => {
          const taskIdx = ep.task_index !== undefined ? Number(ep.task_index) : idx;
          const taskData = tasksData[taskIdx];
          const taskName = taskData?.__index_level_0__ || taskData?.task || `Episode ${idx}`;
          return {
            id: Number(ep.episode_index ?? idx),
            task: taskName,
          };
        });
        setEpisodeInfos(infos);
      } catch (err) {
        console.log("Could not load episode metadata for sidebar");
      }
    };
    loadEpisodeMetadata();
  }, []);

  // Filter episodes based on search
  const filteredEpisodes = episodeInfos.filter(
    (ep) =>
      ep.task.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ep.id.toString().includes(searchQuery)
  );

  // Get short task label (extract force level)
  const getShortLabel = (task: string): string => {
    // Extract "XN" pattern from task string
    const match = task.match(/(\d+N)/);
    return match ? match[1] : `Ep ${task.slice(0, 10)}`;
  };

  React.useEffect(() => {
    if (!sidebarVisible) return;
    function handleClickOutside(event: MouseEvent) {
      if (
        sidebarRef.current &&
        !sidebarRef.current.contains(event.target as Node)
      ) {
        setTimeout(() => setSidebarVisible(false), 500);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [sidebarVisible]);

  return (
    <div className="flex z-10 min-h-screen absolute md:static" ref={sidebarRef}>
      <nav
        className={`shrink-0 overflow-y-auto bg-slate-900 p-5 break-words md:max-h-screen w-72 md:shrink ${!sidebarVisible ? "hidden" : ""
          }`}
        aria-label="Sidebar navigation"
      >
        {/* Dataset Info */}
        <div className="mb-4 text-sm text-slate-400">
          <div>Episodes: {datasetInfo.total_episodes}</div>
          <div>Frames: {datasetInfo.total_frames.toLocaleString()}</div>
          <div>FPS: {datasetInfo.fps}</div>
        </div>

        {/* Quick Jump Dropdown */}
        <div className="mb-4">
          <label className="text-xs text-slate-500 uppercase tracking-wide">Quick Jump</label>
          <div className="relative mt-1">
            <input
              type="text"
              placeholder="Search by force (e.g., 10N)"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setShowDropdown(true);
              }}
              onFocus={() => setShowDropdown(true)}
              className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-sm text-white placeholder-slate-500 focus:outline-none focus:border-sky-500"
            />
            {showDropdown && searchQuery && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-slate-800 border border-slate-700 rounded max-h-48 overflow-y-auto z-20">
                {filteredEpisodes.length > 0 ? (
                  filteredEpisodes.map((ep) => (
                    <Link
                      key={ep.id}
                      href={`/local/frank/episode_${ep.id}`}
                      onClick={() => {
                        setShowDropdown(false);
                        setSearchQuery("");
                      }}
                      className={`block px-3 py-2 text-sm hover:bg-slate-700 ${ep.id === episodeId ? "bg-sky-900 text-sky-300" : "text-slate-300"
                        }`}
                    >
                      <span className="font-semibold">{getShortLabel(ep.task)}</span>
                      <span className="ml-2 text-slate-500">Ep {ep.id}</span>
                    </Link>
                  ))
                ) : (
                  <div className="px-3 py-2 text-sm text-slate-500">No matches</div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Episode List with Task Names */}
        <div className="mb-2 text-xs text-slate-500 uppercase tracking-wide">Episodes</div>
        <div className="space-y-1">
          {episodeInfos.length > 0 ? (
            episodeInfos.map((ep) => (
              <Link
                key={ep.id}
                href={`/local/frank/episode_${ep.id}`}
                className={`block px-3 py-2 rounded text-sm transition-colors ${ep.id === episodeId
                  ? "bg-sky-600 text-white font-semibold"
                  : "text-slate-300 hover:bg-slate-800"
                  }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{getShortLabel(ep.task)}</span>
                  <span className="text-xs text-slate-500">#{ep.id}</span>
                </div>
                <div className="text-xs text-slate-400 truncate mt-0.5">
                  {ep.task.length > 35 ? ep.task.slice(0, 35) + "..." : ep.task}
                </div>
              </Link>
            ))
          ) : (
            // Fallback to paginated episodes if metadata not loaded
            paginatedEpisodes.map((episode) => (
              <Link
                key={episode}
                href={`/local/frank/episode_${episode}`}
                className={`block px-3 py-2 rounded text-sm ${episode === episodeId
                  ? "bg-sky-600 text-white font-semibold"
                  : "text-slate-300 hover:bg-slate-800"
                  }`}
              >
                Episode {episode}
              </Link>
            ))
          )}
        </div>

        {/* Pagination (if many episodes and metadata not loaded) */}
        {totalPages > 1 && episodeInfos.length === 0 && (
          <div className="mt-3 flex items-center justify-center text-xs">
            <button
              onClick={prevPage}
              className={`mr-2 rounded bg-slate-800 px-2 py-1 ${currentPage === 1 ? "cursor-not-allowed opacity-50" : ""
                }`}
              disabled={currentPage === 1}
            >
              « Prev
            </button>
            <span className="mr-2 font-mono">
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={nextPage}
              className={`rounded bg-slate-800 px-2 py-1 ${currentPage === totalPages
                ? "cursor-not-allowed opacity-50"
                : ""
                }`}
              disabled={currentPage === totalPages}
            >
              Next »
            </button>
          </div>
        )}
      </nav>

      {/* Toggle sidebar button */}
      <button
        className="mx-1 flex items-center opacity-50 hover:opacity-100 focus:outline-none focus:ring-0"
        onClick={toggleSidebar}
        title="Toggle sidebar"
      >
        <div className="h-10 w-2 rounded-full bg-slate-500"></div>
      </button>
    </div>
  );
};

export default Sidebar;
