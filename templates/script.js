const canvas = document.querySelector("canvas");
const context = canvas.getContext("2d");
const video = document.querySelector("#myVidPlayer");
const socket = new WebSocket(
  "wss://api.hume.ai/v0/stream/models?apiKey=IgSfnRwS4whHOTGMhxwxh5m6CGoKMj8zu1RrzwwoaErcjZkY"
);
const actx = new AudioContext();
const gain = new GainNode(actx);
const oscs = [new OscillatorNode(actx, { type: "sine", frequency: 440 })];
gain.connect(actx.destination);

oscs[0].connect(gain);
oscs[0].start();

socket.onopen = (e) => console.log("Successfully connected to HUME");
socket.onerror = (e) => console.error(`Error connecting to HUME api: ${e}`);
socket.onmessage = async (e) => {
  actx.resume();
  const data = JSON.parse(e.data);
  const faces = data["face"];
  if (faces["code"] == "W0103") {
    console.log("No faces detected");
  } else {
    const emotions = faces["predictions"][0]["emotions"];
    const result = await fetch("/freq", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(emotions),
    });
    console.log(result.json());
  }
};

var w, h;
canvas.style.display = "none";

async function snapshot() {
  context.fillRect(0, 0, w, h);
  context.drawImage(video, 0, 0, w, h);
  const snapBlob = await canvasToImageBlob(canvas);
  const blob64 = await blobToBase64(snapBlob);
  const requestData = JSON.stringify({
    data: blob64,
    models: {
      face: {},
    },
  });

  if (socket.readyState === WebSocket.OPEN) {
    socket.send(requestData);
  }
}

window.navigator.mediaDevices
  .getUserMedia({ video: true, audio: true })
  .then((stream) => {
    video.srcObject = stream;
    video.onloadedmetadata = (e) => {
      video.play();

      w = video.videoWidth;
      h = video.videoHeight;

      canvas.width = w;
      canvas.height = h;
      snapshot();
    };
  })
  .catch((error) => {
    alert("You have to enable the microphone and the camera");
  });
setInterval(() => snapshot(), 1000);


/**
 * Mute Button
 */
const muteButton = document.querySelector("#muteButton");

// Initially set to muted
let isMuted = true;

muteButton.addEventListener("click", function() {
  if (isMuted) {
    video.muted = false;
    muteButton.innerText = "Unmute";
    isMuted = false;
  } else {
    video.muted = true;
    muteButton.innerText = "Mute";
    isMuted = true;
  }
});