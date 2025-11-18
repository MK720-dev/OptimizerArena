import axios from "axios";
import type { TrainRequest, TrainResponse, PCARequest, PCAResponse } from "../types/backend"

export async function testConnection(): Promise<string> {
    try {
        const res = await axios.get("/api/");
        return res.data.message;
    } catch (err) {
        console.error("Connection failed:", err);
        throw err;
    }
}

export async function runTraining(req: TrainRequest): Promise<TrainResponse> {
    const response = await axios.post<TrainResponse>("/api/train", req);
    return response.data;
}

export async function runPCA(req: PCARequest): Promise<PCAResponse> {
    const response = await axios.post<PCAResponse>("/api/pca", req);
    return response.data
}