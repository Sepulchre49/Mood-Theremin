import createSocket from "./socket.js";
import sendSnapshot from "./snapshot.js";
import init from "./script.js";

const canvas = document.querySelector("#imageCapture");
const actx = new AudioContext();
actx.resume();
const gain = new GainNode(actx);
const oscs = [new OscillatorNode(actx, {type:"sine", frequency:440})];
gain.connect(actx.destination);

oscs[0].connect(gain);
oscs[0].start();


//w-width,h-height
var w, h;
canvas.style.display = "none";
const socket = createSocket();

const video = await init();
setInterval(() => sendSnapshot(canvas, video, socket,w,h), 1000);
