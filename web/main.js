import * as ort from 'onnxruntime-web';

function fetchelementValue(x) {
	return parseFloat(document.getElementById(`dim${x}`).value)
}


var session = null;

async function main() {
	try {
		const dim7input = [];
		for (let i = 1; i<=7; i++) {
			dim7input.push(fetchelementValue(i));
		}

		const datainput = Float32Array.from(dim7input);

		const tensordata = new ort.Tensor('float32', datainput, [1,7,1,1]);
		console.log(tensordata);

		session = await ort.InferenceSession.create('./model.onnx');

		const feeds = {'onnx::ConvTranspose_0' : tensordata};
		const results = await session.run(feeds);

		console.log(results);
	} catch (e) {
		console.log(e);
	}
}

// add mutation observer

main();
