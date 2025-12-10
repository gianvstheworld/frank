"use client";
import { useEffect, useState, Suspense } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { parquetReadObjects } from "hyparquet";
import SharedHeader from "@/components/shared-header";

// Color palette for episodes
const EPISODE_COLORS = [
  "#38bdf8", // sky
  "#34d399", // emerald
  "#fbbf24", // amber
  "#f97316", // orange
  "#a78bfa", // violet
  "#f472b6", // pink
  "#22d3ee", // cyan
  "#84cc16", // lime
];

interface DatasetInfo {
  total_episodes: number;
  total_frames: number;
  fps: number;
  robot_type?: string;
}

interface EpisodeMetadata {
  episode_index: number;
  length: number;
  task_index?: number;
}

interface TaskMetadata {
  task_index?: number;
  task?: string;
  __index_level_0__?: string;
}

export default function Home() {
  return (
    <Suspense fallback={null}>
      <HomeInner />
    </Suspense>
  );
}

function HomeInner() {
  const router = useRouter();
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [episodes, setEpisodes] = useState<Array<{ id: number; task: string; frames: number }>>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const episodesPerPage = 8;

  // Hugging Face dataset configuration
  const HF_REPO = "tommaselli/frank_load_experiments";
  const HF_BASE = `https://huggingface.co/datasets/${HF_REPO}/resolve/main`;

  useEffect(() => {
    const loadData = async () => {
      try {
        // Load dataset info from Hugging Face
        const infoRes = await fetch(`${HF_BASE}/meta/info.json`);
        const info: DatasetInfo = await infoRes.json();
        setDatasetInfo(info);

        // Try to load episodes metadata from parquet
        let episodeList: Array<{ id: number; task: string; frames: number }> = [];

        try {
          // Load tasks parquet to get task names
          const tasksRes = await fetch(`${HF_BASE}/meta/tasks.parquet`);
          const tasksBuffer = await tasksRes.arrayBuffer();
          const tasksData = await parquetReadObjects({ file: tasksBuffer }) as TaskMetadata[];

          // Load episodes metadata
          const episodesRes = await fetch(`${HF_BASE}/meta/episodes/chunk-000/file-000.parquet`);
          const episodesBuffer = await episodesRes.arrayBuffer();
          const episodesData = await parquetReadObjects({ file: episodesBuffer }) as EpisodeMetadata[];

          // Map episodes to their tasks
          episodeList = episodesData.map((ep, idx) => {
            const taskIdx = ep.task_index !== undefined ? Number(ep.task_index) : idx;
            const taskData = tasksData[taskIdx];
            // Task name is stored in __index_level_0__ field in LeRobot v3
            const taskName = taskData?.__index_level_0__ || taskData?.task || `Episode ${idx}`;
            return {
              id: Number(ep.episode_index ?? idx),
              task: taskName,
              frames: Number(ep.length || 1000),
            };
          });
        } catch (err) {
          console.log("Could not load episode metadata, using defaults");
          // Fallback: generate episodes from total_episodes
          episodeList = Array.from({ length: info.total_episodes }, (_, i) => ({
            id: i,
            task: `Episode ${i}`,
            frames: Math.floor(info.total_frames / info.total_episodes),
          }));
        }

        setEpisodes(episodeList);
        setLoading(false);
      } catch (err) {
        setError("Failed to load dataset from Hugging Face");
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Filter episodes based on search
  const filteredEpisodes = episodes.filter(ep =>
    ep.task.toLowerCase().includes(searchQuery.toLowerCase()) ||
    ep.id.toString().includes(searchQuery)
  );

  // Paginate
  const totalPages = Math.ceil(filteredEpisodes.length / episodesPerPage);
  const startIdx = (currentPage - 1) * episodesPerPage;
  const paginatedEpisodes = filteredEpisodes.slice(startIdx, startIdx + episodesPerPage);

  return (
    <div className="min-h-screen">
      {/* Header */}
      <SharedHeader />

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-white mb-4">
            Dynamic Performance Under Load Variation
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Analyze robot dynamics with different external forces applied to the end-effector.
            Compare tracking performance, torques, and errors across load conditions.
          </p>
        </div>

        {/* Hero Video */}
        <div className="mb-12 max-w-3xl mx-auto">
          <div className="rounded-xl overflow-hidden shadow-2xl shadow-sky-500/10 border border-slate-700">
            <video
              autoPlay
              loop
              muted
              playsInline
              className="w-full"
            >
              <source src={`${process.env.NEXT_PUBLIC_BASE_PATH || ''}/robot-demo.mp4`} type="video/mp4" />
            </video>
          </div>
          <p className="text-center text-slate-500 text-sm mt-3">
            FRANK robot performing sinusoidal trajectory tracking
          </p>
        </div>

        {/* Dataset Info */}
        {datasetInfo && (
          <div className="grid grid-cols-2 gap-4 mb-12 max-w-md mx-auto">
            <InfoCard label="Episodes" value={datasetInfo.total_episodes} />
            <InfoCard label="Total Frames" value={datasetInfo.total_frames?.toLocaleString()} />
          </div>
        )}

        {/* Episode Cards */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold text-white">Select Episode</h3>
          <input
            type="text"
            placeholder="Search by force (e.g., 10N)"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
            className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-sky-500 w-64"
          />
        </div>

        {loading ? (
          <div className="text-center text-slate-400 py-12">Loading episodes...</div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
              {paginatedEpisodes.map((episode, idx) => (
                <EpisodeCard
                  key={episode.id}
                  episode={episode}
                  color={EPISODE_COLORS[(startIdx + idx) % EPISODE_COLORS.length]}
                  onClick={() => router.push(`/local/frank/episode_${episode.id}`)}
                />
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-4 mb-12">
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className={`px-4 py-2 rounded-lg text-sm ${currentPage === 1 ? "bg-slate-800 text-slate-500 cursor-not-allowed" : "bg-slate-700 text-white hover:bg-slate-600"}`}
                >
                  ← Previous
                </button>
                <span className="text-slate-400 text-sm">
                  Page {currentPage} of {totalPages} ({filteredEpisodes.length} episodes)
                </span>
                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className={`px-4 py-2 rounded-lg text-sm ${currentPage === totalPages ? "bg-slate-800 text-slate-500 cursor-not-allowed" : "bg-slate-700 text-white hover:bg-slate-600"}`}
                >
                  Next →
                </button>
              </div>
            )}

            {filteredEpisodes.length === 0 && (
              <div className="text-center text-slate-400 py-8 mb-12">
                No episodes match "{searchQuery}"
              </div>
            )}
          </>
        )}

        {/* Features */}
        <div className="mt-16 text-center">
          <h3 className="text-xl font-semibold text-white mb-6">Visualization Features</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <FeatureCard
              icon="📊"
              title="Time Series Charts"
              description="Interactive plots for observation state, actions, and forces"
            />
            <FeatureCard
              icon="🎚️"
              title="Synchronized Playback"
              description="Scrub through the data timeline with synchronized chart cursors"
            />
            <FeatureCard
              icon="📈"
              title="Performance Metrics"
              description="Analyze observation states, actions, and external forces"
            />
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-8 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-center">
            {error}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-700/50 mt-16">
        <div className="max-w-6xl mx-auto px-6 py-6 text-center text-slate-500 text-sm">
          Robotic Manipulation Class | Second Semester 2025
        </div>
      </footer>
    </div>
  );
}

function InfoCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-xs text-slate-400 uppercase tracking-wide">{label}</div>
    </div>
  );
}

function EpisodeCard({ episode, color, onClick }: {
  episode: { id: number; task: string; frames: number };
  color: string;
  onClick: () => void;
}) {
  // Extract force level from task string for prominent display
  const forceMatch = episode.task.match(/(\d+)N/);
  const forceLabel = forceMatch ? `${forceMatch[1]}N` : `Ep ${episode.id}`;

  return (
    <button
      onClick={onClick}
      className="group bg-slate-800/50 border border-slate-700/50 rounded-xl p-5 text-left transition-all hover:bg-slate-800 hover:border-slate-600 hover:shadow-lg"
    >
      <div className="flex items-center gap-3 mb-3">
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm"
          style={{ backgroundColor: color }}
        >
          {forceLabel}
        </div>
        <span className="text-slate-400 text-sm">Episode {episode.id}</span>
      </div>
      <p className="text-slate-300 text-sm line-clamp-2">{episode.task}</p>
      <div className="mt-3 text-xs text-slate-500">{episode.frames.toLocaleString()} frames</div>
    </button>
  );
}

function FeatureCard({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <div className="bg-slate-800/30 rounded-lg p-6 border border-slate-700/30">
      <div className="text-3xl mb-3">{icon}</div>
      <h4 className="text-white font-semibold mb-2">{title}</h4>
      <p className="text-slate-400 text-sm">{description}</p>
    </div>
  );
}
