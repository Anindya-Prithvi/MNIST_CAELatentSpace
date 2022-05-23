import * as ort from 'onnxruntime-web';


for (let i = 1; i <= 7; i++) {
	const foundelement = document.getElementById(`dim${i}`);
	foundelement.addEventListener('input', () => {
		// console.log(`the value ${foundelement.value} for elem ${i}`);
		overwriteCanvas();
	});
}

function fetchelementValue(x) {
	const foundelement = document.getElementById(`dim${x}`);
	return parseFloat(foundelement.value);
}

async function overwriteCanvas() {
	var canvas = document.getElementById("displayHS");
	var ctx = canvas.getContext("2d");

	var imgData = ctx.getImageData(0, 0, 28, 28);
	var data = imgData.data;

	const dim7input = [];

	for (let i = 1; i <= 7; i++) {
		dim7input.push(fetchelementValue(i));
	}

	const datainput = Float32Array.from(dim7input);

	const tensordata = new ort.Tensor('float32', datainput, [1, 7, 1, 1]);
	const feeds = { 'onnx::ConvTranspose_0': tensordata };
	const res = await session.run(feeds);

	var output = res[31].reshape([28, 28]).data;

	for (var i = 0; i < data.length; i += 4) {
		var curindex = ~~(i / 4);
		data[i] = Math.round(output[curindex] * 255) % 255;
		data[i + 1] = Math.round(output[curindex] * 255) % 255;
		data[i + 2] = Math.round(output[curindex] * 255) % 255;
		data[i + 3] = 255;
	}

	ctx.putImageData(imgData, 0, 0);

	// create a new img object
	var image = document.getElementById('maindisplay');

	// set the img.src to the canvas data url
	image.src = canvas.toDataURL();
	// console.log(image.src);

	// append the new img object to the page
	// document.body.appendChild(image);

}


var session = null;

async function main() {
	try {
		// const dim7input = [];
		// for (let i = 1; i <= 7; i++) {
		// 	dim7input.push(fetchelementValue(i));
		// }

		// const datainput = Float32Array.from(dim7input);

		// const tensordata = new ort.Tensor('float32', datainput, [1,7,1,1]);
		// console.log(tensordata);

		session = await ort.InferenceSession.create('./model.onnx');

		// const feeds = { 'onnx::ConvTranspose_0': tensordata };
		// const results = await session.run(feeds);

		// console.log(results);
	} catch (e) {
		console.log(e);
	}
}

// add mutation observer

main();
