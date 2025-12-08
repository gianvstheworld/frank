/**
 * Utility functions for fetching dataset from Hugging Face Hub
 * Configured for: tommaselli/frank_load_experiments
 */

// Hugging Face dataset configuration
const HF_DATASET_REPO = "tommaselli/frank_load_experiments";
const HF_DATASET_BRANCH = "main";

/**
 * Builds the Hugging Face Hub URL for a dataset file
 * Always uses the configured HF_DATASET_REPO, ignoring the repoId parameter
 */
function getHuggingFaceUrl(repoId: string, path: string): string {
  // Always use the configured HF dataset, regardless of what repoId is passed
  return `https://huggingface.co/datasets/${HF_DATASET_REPO}/resolve/${HF_DATASET_BRANCH}/${path}`;
}

/**
 * Dataset information structure from info.json
 */
interface DatasetInfo {
  codebase_version: string;
  robot_type: string | null;
  total_episodes: number;
  total_frames: number;
  total_tasks: number;
  chunks_size: number;
  data_files_size_in_mb: number;
  video_files_size_in_mb: number;
  fps: number;
  splits: Record<string, string>;
  data_path: string;
  video_path: string | null;
  features: Record<string, any>;
}

/**
 * Fetches dataset information from Hugging Face Hub
 */
export async function getDatasetInfo(repoId: string): Promise<DatasetInfo> {
  try {
    const url = getHuggingFaceUrl(repoId, "meta/info.json");

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);

    const response = await fetch(url, {
      method: "GET",
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Failed to fetch dataset info from Hugging Face: ${response.status}`);
    }

    const data = await response.json();

    if (!data.features) {
      throw new Error("Dataset info.json does not have the expected features structure");
    }

    return data as DatasetInfo;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(
      `Failed to load dataset from Hugging Face. ` +
      "Please check that the dataset exists and is accessible."
    );
  }
}


/**
 * Gets the dataset version by reading the codebase_version
 */
export async function getDatasetVersion(repoId: string): Promise<string> {
  try {
    const datasetInfo = await getDatasetInfo(repoId);

    const codebaseVersion = datasetInfo.codebase_version;
    if (!codebaseVersion) {
      throw new Error("Dataset info.json does not contain codebase_version");
    }

    // Validate that it's a supported version
    const supportedVersions = ["v3.0", "v2.1", "v2.0"];
    if (!supportedVersions.includes(codebaseVersion)) {
      throw new Error(
        `Dataset has codebase version ${codebaseVersion}, which is not supported. ` +
        "This tool only works with dataset versions 3.0, 2.1, or 2.0."
      );
    }

    return codebaseVersion;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(
      `Dataset is not compatible with this visualizer.`
    );
  }
}

/**
 * Builds URL for Hugging Face dataset files
 */
export function buildVersionedUrl(repoId: string, version: string, path: string): string {
  return getHuggingFaceUrl(repoId, path);
}
