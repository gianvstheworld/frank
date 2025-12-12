import EpisodeViewer from "./episode-viewer";
import { getEpisodeDataSafe } from "./fetch-data";
import { Suspense } from "react";

// Generate static paths for FRANK robot episodes
// Add more episodes here if you record additional experiments
// The episode IDs should match your dataset's episode_index values
export async function generateStaticParams() {
  // Generate params for episodes 0-49 (extend if needed for more)
  const maxEpisodes = 50;
  return Array.from({ length: maxEpisodes }, (_, i) => ({
    org: "local",
    dataset: "frank",
    episode: `episode_${i}`,
  }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ org: string; dataset: string; episode: string }>;
}) {
  const { org, dataset, episode } = await params;
  return {
    title: `${org}/${dataset} | episode ${episode}`,
  };
}

export default async function EpisodePage({
  params,
}: {
  params: Promise<{ org: string; dataset: string; episode: string }>;
}) {
  // episode is like 'episode_1'
  const { org, dataset, episode } = await params;
  // fetchData should be updated if needed to support this path pattern
  const episodeNumber = Number(episode.replace(/^episode_/, ""));
  const { data, error } = await getEpisodeDataSafe(org, dataset, episodeNumber);
  return (
    <Suspense fallback={null}>
      <EpisodeViewer data={data} error={error} />
    </Suspense>
  );
}
