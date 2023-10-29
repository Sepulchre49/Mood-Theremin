const canvas = document.querySelector("canvas");
const context = canvas.getContext("2d");
const video = document.querySelector("#myVidPlayer");
const socket = new WebSocket(
  "wss://api.hume.ai/v0/stream/models?apiKey=IgSfnRwS4whHOTGMhxwxh5m6CGoKMj8zu1RrzwwoaErcjZkY"
);
const actx = new AudioContext();
const gain = new GainNode(actx);
const oscs = [new OscillatorNode(actx), new OscillatorNode(actx), new OscillatorNode(actx), new OscillatorNode(actx)];
oscs.forEach(osc => {
    osc.start();
    osc.connect(gain);
});
gain.connect(actx.destination);

socket.onopen = e => console.log("Successfully connected to HUME");
socket.onerror = e => console.error(`Error connecting to HUME api: ${e}`);
socket.onmessage = async e => {
    actx.resume();
    const data = JSON.parse(e.data);
    //console.log(JSON.stringify(data));
    //const emotions = data["face"]["predictions"][0]["emotions"].sort((a,b) => b.score - a.score);
    const faces = data["face"];
    if (faces["code"] == "W0103") {
        console.log("No faces detected");
    } else {
        const emotions = faces["predictions"][0]["emotions"];
        const result = await fetch("/freq", {
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": JSON.stringify(emotions)
        });
        results = await result.json();
        results = results[0]
        //console.log(results);
        gain.gain.exponentialRampToValueAtTime(results[4], 0.3);
        console.log(gain.gain.value)
        for (let i =0; i<4; i++) {
            oscs[i].frequency.exponentialRampToValueAtTime(results[i],0.3);
        }
        console.log(oscs);
    }
}

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
    muteButton.innerText = "Unmute";
    isMuted = false;
  } else {
    muteButton.innerText = "Mute";
    isMuted = true;
  }
});

// TODO Have mute button save prev amplitude value