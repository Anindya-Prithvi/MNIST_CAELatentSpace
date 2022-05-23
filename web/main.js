import * as ort from 'onnxruntime-web';

async function main() {
	try {
		const datainput = Float32Array.from([1,2,3,4,5,6,7]);

		const tensordata = new ort.Tensor('float32', datainput, [1,7,1,1]);
		console.log(tensordata);

		const session = await ort.InferenceSession.create('./model.onnx');

		const feeds = {'onnx::ConvTranspose_0' : tensordata};
		const results = await session.run(feeds);

		console.log(results);
	} catch (e) {
		console.log(e);
	}
}

main();
