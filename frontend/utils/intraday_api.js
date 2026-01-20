// Intraday API functions

const API_BASE_URL = 'http://localhost:8000/api/v1';

export async function fetchIntradaySignals() {
    try {
        const response = await fetch(`${API_BASE_URL}/intraday/signals`);
        if (!response.ok) {
            throw new Error('Failed to fetch intraday signals');
        }
        return await response.json();
    } catch (error) {
        console.error('Intraday API Error:', error);
        throw error;
    }
}

export async function fetchIntradayStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/intraday/stats`);
        if (!response.ok) {
            throw new Error('Failed to fetch intraday stats');
        }
        return await response.json();
    } catch (error) {
        console.error('Intraday Stats API Error:', error);
        throw error;
    }
}
