export default function createSocket() {
    const socket = new WebSocket("wss://api.hume.ai/v0/stream/models?apiKey=IgSfnRwS4whHOTGMhxwxh5m6CGoKMj8zu1RrzwwoaErcjZkY")

    socket.onopen = e => console.log("Successfully connected to HUME");

    socket.onerror = e => console.error(`Error connecting to HUME api: ${e}`);

    socket.onmessage = e => {
        const data = JSON.parse(e.data);
        if (data["face"]["code"] == "W0103") {
            console.log(data["face"]["warning"])
        } else {
            const emotions = data["face"]["predictions"][0]["emotions"].sort((a,b) => b.score - a.score);
            const topScore = emotions[0].score;
            gain.gain.value = topScore;
            console.log(gain.gain.value);
        }
    }
    return socket;
}