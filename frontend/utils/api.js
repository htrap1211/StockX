const API_BASE_URL = "http://localhost:8000/api/v1";

export async function fetchRecommendations(market = "US") {
    try {
        const res = await fetch(`${API_BASE_URL}/recommendations?market=${market}`, {
            cache: "no-store",
        });
        if (!res.ok) {
            throw new Error("Failed to fetch data");
        }
        return res.json();
    } catch (error) {
        console.error("API Error:", error);
        return [];
    }
}
