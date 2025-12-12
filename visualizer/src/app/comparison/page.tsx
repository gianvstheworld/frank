"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import SharedHeader from "@/components/shared-header";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    LineChart,
    Line,
    Label,
} from "recharts";

interface ComparisonData {
    "Force (N)": number;
    "RMS Error (rad)": number;
    "Peak Torque (Nm)": number;
    "Mean Torque (Nm)": number;
    "Overshoot (%)": number;
    "Settling Time (s)": number;
    "Rise Time (s)": number;
    "SS Error (rad)": number;
    IAE: number;
    "Max Vel Error (rad/s)": number;
}

interface MetricsData {
    metrics: any[];
    comparison: ComparisonData[];
}

// Custom tooltip formatter with limited decimals
const formatValue = (value: number, decimals: number = 3) => {
    return Number(value).toFixed(decimals);
};

const CustomTooltip = ({ active, payload, label, unit, decimals = 3 }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 shadow-lg">
                <p className="text-slate-300 font-medium mb-1">{label}</p>
                {payload.map((entry: any, index: number) => (
                    <p key={index} className="text-sm" style={{ color: entry.color }}>
                        {entry.name}: {formatValue(entry.value, decimals)} {unit}
                    </p>
                ))}
            </div>
        );
    }
    return null;
};

export default function ComparisonPage() {
    const [data, setData] = useState<MetricsData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Hugging Face dataset configuration
    const HF_REPO = "tommaselli/frank_load_experiments";
    const HF_BASE = `https://huggingface.co/datasets/${HF_REPO}/resolve/main`;

    useEffect(() => {
        fetch(`${HF_BASE}/metrics.json`)
            .then((res) => res.json())
            .then((data) => {
                setData(data);
                setLoading(false);
            })
            .catch((err) => {
                setError("Failed to load metrics data from Hugging Face. Make sure metrics.json is uploaded.");
                setLoading(false);
            });
    }, []);

    if (loading) {
        return (
            <div className="min-h-screen bg-slate-950 flex items-center justify-center">
                <div className="text-white">Loading metrics...</div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="min-h-screen bg-slate-950 flex items-center justify-center">
                <div className="text-red-400">{error || "No data available"}</div>
            </div>
        );
    }

    // Prepare chart data - aggregate by force level
    const chartData = data.comparison.reduce((acc: any[], item) => {
        const existing = acc.find((d) => d.force === item["Force (N)"]);
        if (existing) {
            // Average duplicate force levels
            existing.rmsError = (existing.rmsError + item["RMS Error (rad)"]) / 2;
            existing.peakTorque = (existing.peakTorque + item["Peak Torque (Nm)"]) / 2;
            existing.meanTorque = (existing.meanTorque + item["Mean Torque (Nm)"]) / 2;
            existing.overshoot = (existing.overshoot + item["Overshoot (%)"]) / 2;
            existing.iae = (existing.iae + item["IAE"]) / 2;
        } else {
            acc.push({
                force: item["Force (N)"],
                label: `${item["Force (N)"]}N`,
                rmsError: item["RMS Error (rad)"],
                peakTorque: item["Peak Torque (Nm)"],
                meanTorque: item["Mean Torque (Nm)"],
                overshoot: item["Overshoot (%)"],
                settlingTime: item["Settling Time (s)"],
                iae: item["IAE"],
                maxVelError: item["Max Vel Error (rad/s)"],
            });
        }
        return acc;
    }, []);

    return (
        <div className="min-h-screen">
            {/* Header */}
            <SharedHeader />

            <main className="max-w-7xl mx-auto px-6 py-8">
                <h2 className="text-3xl font-bold text-white mb-2">
                    Force Comparison Analysis
                </h2>
                <p className="text-slate-400 mb-8">
                    Compare performance metrics across different external force conditions
                </p>

                {/* Summary Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    <MetricCard
                        label="Force Levels"
                        value={`${chartData.length}`}
                        unit="conditions"
                    />
                    <MetricCard
                        label="Min RMS Error"
                        value={(Math.min(...chartData.map((d) => d.rmsError)) * 1000).toFixed(2)}
                        unit="mrad"
                    />
                    <MetricCard
                        label="Max Overshoot"
                        value={Math.max(...chartData.map((d) => d.overshoot)).toFixed(1)}
                        unit="%"
                    />
                    <MetricCard
                        label="Avg Mean Torque"
                        value={(chartData.reduce((s, d) => s + d.meanTorque, 0) / chartData.length).toFixed(1)}
                        unit="Nm"
                    />
                </div>

                {/* Charts Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    {/* RMS Error Chart */}
                    <ChartCard title="RMS Error vs Force">
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={chartData} margin={{ bottom: 30, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="label" stroke="#94a3b8">
                                    <Label value="External Force" position="bottom" offset={10} fill="#94a3b8" />
                                </XAxis>
                                <YAxis stroke="#94a3b8">
                                    <Label value="RMS Error (rad)" angle={-90} position="insideLeft" fill="#94a3b8" style={{ textAnchor: 'middle' }} />
                                </YAxis>
                                <Tooltip content={<CustomTooltip unit="rad" decimals={4} />} />
                                <Bar dataKey="rmsError" fill="#38bdf8" name="RMS Error" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartCard>

                    {/* Overshoot Chart */}
                    <ChartCard title="Overshoot vs Force">
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={chartData} margin={{ bottom: 30, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="label" stroke="#94a3b8">
                                    <Label value="External Force" position="bottom" offset={10} fill="#94a3b8" />
                                </XAxis>
                                <YAxis stroke="#94a3b8">
                                    <Label value="Overshoot (%)" angle={-90} position="insideLeft" fill="#94a3b8" style={{ textAnchor: 'middle' }} />
                                </YAxis>
                                <Tooltip content={<CustomTooltip unit="%" decimals={1} />} />
                                <Bar dataKey="overshoot" fill="#f97316" name="Overshoot" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartCard>

                    {/* Mean Torque Chart */}
                    <ChartCard title="Mean Torque vs Force">
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={chartData} margin={{ bottom: 30, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="label" stroke="#94a3b8">
                                    <Label value="External Force" position="bottom" offset={10} fill="#94a3b8" />
                                </XAxis>
                                <YAxis stroke="#94a3b8">
                                    <Label value="Mean Torque (Nm)" angle={-90} position="insideLeft" fill="#94a3b8" style={{ textAnchor: 'middle' }} />
                                </YAxis>
                                <Tooltip content={<CustomTooltip unit="Nm" decimals={2} />} />
                                <Bar dataKey="meanTorque" fill="#34d399" name="Mean Torque" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartCard>

                    {/* IAE Chart */}
                    <ChartCard title="Integral Absolute Error vs Force">
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={chartData} margin={{ bottom: 30, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="label" stroke="#94a3b8">
                                    <Label value="External Force" position="bottom" offset={10} fill="#94a3b8" />
                                </XAxis>
                                <YAxis stroke="#94a3b8">
                                    <Label value="IAE" angle={-90} position="insideLeft" fill="#94a3b8" style={{ textAnchor: 'middle' }} />
                                </YAxis>
                                <Tooltip content={<CustomTooltip unit="" decimals={4} />} />
                                <Bar dataKey="iae" fill="#a78bfa" name="IAE" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartCard>
                </div>

                {/* Data Table */}
                <div className="bg-slate-800/50 rounded-lg border border-slate-700/50 overflow-hidden">
                    <h3 className="text-lg font-semibold text-white px-6 py-4 border-b border-slate-700/50">
                        Detailed Metrics
                    </h3>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead className="bg-slate-700/50">
                                <tr>
                                    <th className="px-4 py-3 text-left text-slate-300">Force</th>
                                    <th className="px-4 py-3 text-right text-slate-300">RMS Error</th>
                                    <th className="px-4 py-3 text-right text-slate-300">Mean Torque</th>
                                    <th className="px-4 py-3 text-right text-slate-300">Overshoot</th>
                                    <th className="px-4 py-3 text-right text-slate-300">Settling Time</th>
                                    <th className="px-4 py-3 text-right text-slate-300">IAE</th>
                                </tr>
                            </thead>
                            <tbody>
                                {chartData.map((row, idx) => (
                                    <tr
                                        key={idx}
                                        className="border-t border-slate-700/50 hover:bg-slate-700/30"
                                    >
                                        <td className="px-4 py-3 font-semibold text-white">
                                            {row.label}
                                        </td>
                                        <td className="px-4 py-3 text-right text-slate-300">
                                            {(row.rmsError * 1000).toFixed(2)} mrad
                                        </td>
                                        <td className="px-4 py-3 text-right text-slate-300">
                                            {row.meanTorque.toFixed(2)} Nm
                                        </td>
                                        <td className="px-4 py-3 text-right text-slate-300">
                                            {row.overshoot.toFixed(1)}%
                                        </td>
                                        <td className="px-4 py-3 text-right text-slate-300">
                                            {row.settlingTime.toFixed(2)} s
                                        </td>
                                        <td className="px-4 py-3 text-right text-slate-300">
                                            {row.iae.toFixed(4)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </main>
        </div>
    );
}

function MetricCard({
    label,
    value,
    unit,
}: {
    label: string;
    value: string;
    unit: string;
}) {
    return (
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
                {label}
            </div>
            <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold text-white">{value}</span>
                <span className="text-sm text-slate-400">{unit}</span>
            </div>
        </div>
    );
}

function ChartCard({
    title,
    children,
}: {
    title: string;
    children: React.ReactNode;
}) {
    return (
        <div className="bg-slate-800/50 rounded-lg border border-slate-700/50 p-4">
            <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
            {children}
        </div>
    );
}
