function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (reader.result) {
        const result = reader.result.toString() ;
        resolve(result.split(",")[1]);
      }
    };
    reader.readAsDataURL(blob);
  });
}

function canvasToImageBlob(canvas, format = "image/png") {
  return new Promise((resolve, reject) => {
    const handleBlob = (blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject("Could not parse blob");
      }
    };
    canvas.toBlob(handleBlob, format, 1);
  });
}
