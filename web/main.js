import * as ort from 'onnxruntime-web';

async function main() {
	try {
		const session = await ort.InferenceSession.create('./model.onnx');
	} catch (e) {
		console.log(e);
	}
}

main();
