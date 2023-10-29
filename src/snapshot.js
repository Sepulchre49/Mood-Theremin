import { blobToBase64, canvasToImageBlob } from "./utils/blobUtils";

export default async function sendSnapshot(canvas,video,socket,w,h) {
    console.log("making a new request");
    const context = canvas.getContext("2d");
    canvas.width = w;
    canvas.height = h;
    context.fillRect(0, 0, w, h);
    context.drawImage(video, 0, 0, w, h);
    const snapBlob = await canvasToImageBlob(canvas);
    const blob64 = await blobToBase64(snapBlob);

    console.log(blob64);
    const requestData = JSON.stringify({
        data: blob64,
        models: {
            face: {},
        }
    });

    if (socket.readyState === WebSocket.OPEN) {
        socket.send(requestData);
    }
}
